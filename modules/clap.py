import logging
import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
import soundfile as sf
import time
import os
import json

def apply_temporal_nms(events, min_gap_seconds=1.0):
    if not events:
        return []
    sorted_events = sorted(events, key=lambda x: x.get('confidence', 0.0), reverse=True)
    used_times = set()
    filtered_events = []
    for event in sorted_events:
        start_time = event.get('start')
        if start_time is None or not isinstance(start_time, (int, float)):
            continue
        time_window = int(start_time)
        nearby_used = any(t in used_times for t in range(time_window - int(min_gap_seconds) + 1, time_window + int(min_gap_seconds)))
        if not nearby_used:
            filtered_events.append(event)
            used_times.add(time_window)
    return sorted(filtered_events, key=lambda x: x.get('start', 0.0))

def annotate_with_clap(audio_obj, config=None):
    """
    Annotate audio with CLAP event detection. Stores events in audio_obj.metadata['clap_events'].
    Processes audio in short, fixed-length mono chunks (default 10s, max 30s) to avoid OOM errors.
    Configurable via config: 'chunk_duration' (float, seconds, default 10.0, max 30.0), 'min_gap' (float, seconds).
    Also writes detected events to a JSON file in the 'clap' subfolder of the run directory.
    """
    logger = logging.getLogger(__name__)
    audio_path = audio_obj.path
    target_events = config.get('target_events', [
        "telephone ringing", "speech", "conversation", "hang-up tones"
    ])
    threshold = float(config.get('threshold', 0.98))
    chunk_duration = float(config.get('chunk_duration', 10.0))
    min_gap = float(config.get('min_gap', 1.0))
    model_id = config.get('clap_model_id', "laion/clap-htsat-unfused")
    sample_rate = int(config.get('sample_rate', 48000))
    max_chunk_duration = 30.0
    if chunk_duration > max_chunk_duration:
        logger.warning(f"chunk_duration {chunk_duration}s too large, capping to {max_chunk_duration}s.")
        chunk_duration = max_chunk_duration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        audio_data, sr = sf.read(audio_path)
        if sr != sample_rate:
            logger.error(f"CLAP requires {sample_rate}Hz audio, got {sr}Hz.")
            audio_obj.metadata['clap_events'] = {event: [] for event in target_events}
            return audio_obj
        # Convert to mono if needed
        if audio_data.ndim > 1:
            logger.info(f"Input audio has {audio_data.shape[1]} channels, converting to mono.")
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)
        clap_model = ClapModel.from_pretrained(model_id).to(device)
        clap_processor = ClapProcessor.from_pretrained(model_id)
        clap_model.eval()
        # Precompute text features
        with torch.amp.autocast(device_type=device.type):
            text_inputs = clap_processor(text=target_events, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            with torch.no_grad():
                precomputed_text_features = clap_model.get_text_features(**text_inputs)
        # Process audio in chunks
        chunk_size = int(chunk_duration * sample_rate)
        num_chunks = int(np.ceil(len(audio_data) / chunk_size))
        all_detected_events = {event: [] for event in target_events}
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
            logger.debug(f"CLAP chunk {i+1}/{num_chunks}: shape={chunk.shape}, dtype={chunk.dtype}, duration={len(chunk)/sample_rate:.2f}s")
            if len(chunk) < sample_rate * 0.1:
                continue
            if len(chunk) > int(max_chunk_duration * sample_rate):
                logger.error(f"Chunk {i+1} is too large for CLAP: {len(chunk)/sample_rate:.2f}s. Skipping.")
                continue
            # Ensure chunk is 1D float32
            if chunk.ndim != 1 or chunk.dtype != np.float32:
                logger.error(f"Chunk shape/dtype invalid: shape={chunk.shape}, dtype={chunk.dtype}")
                continue
            with torch.amp.autocast(device_type=device.type):
                inputs = clap_processor(audios=[chunk], sampling_rate=sample_rate, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    audio_features = clap_model.get_audio_features(**inputs)
                    similarity = torch.nn.functional.cosine_similarity(
                        audio_features[:, None],
                        precomputed_text_features[None, :],
                        dim=-1
                    )
            matches = torch.where(similarity > threshold)
            for audio_idx, text_idx in zip(*matches):
                chunk_start_time = start_idx / sample_rate
                chunk_end_time = end_idx / sample_rate
                all_detected_events[target_events[text_idx]].append({
                    'type': target_events[text_idx],
                    'confidence': float(similarity[audio_idx, text_idx]),
                    'start': chunk_start_time,
                    'end': chunk_end_time
                })
        # Apply temporal NMS
        final_events = {}
        for event_type, events in all_detected_events.items():
            final_events[event_type] = apply_temporal_nms(events, min_gap_seconds=min_gap)
        audio_obj.metadata['clap_events'] = final_events
        audio_obj.add_provenance('annotate_with_clap', {'events': {k: len(v) for k, v in final_events.items()}})
        logger.info(f"CLAP event detection complete. Events: { {k: len(v) for k, v in final_events.items()} }")
        # Write events to JSON file in clap subfolder
        clap_dir = config.get('stage_dirs', {}).get('clap')
        if clap_dir:
            events_path = os.path.join(clap_dir, 'events.json')
            with open(events_path, 'w') as f:
                json.dump(final_events, f, indent=2)
            logger.info(f"CLAP events written to {events_path}")
        else:
            logger.warning("No clap_dir found in config; CLAP events not saved to disk.")
        # Log event types, counts, and timestamps
        for event_type, events in final_events.items():
            logger.info(f"CLAP event: {event_type} - {len(events)} event(s)")
            for ev in events:
                logger.info(f"  {event_type}: {ev['start']:.2f}s - {ev['end']:.2f}s (conf={ev['confidence']:.2f})")
    except Exception as e:
        logger.error(f"CLAP event detection failed: {e}", exc_info=True)
        audio_obj.metadata['clap_events'] = {event: [] for event in target_events}
    return audio_obj 