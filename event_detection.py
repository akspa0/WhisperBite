import os
import logging
import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
import soundfile as sf
from typing import List, Dict, Optional, Tuple
import time
from tqdm import tqdm

# Constants
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000

# Default event types to detect
DEFAULT_EVENTS = [
    "telephone ringing",
    "speech",
    "conversation",
    "silence",
    "background noise",
    "phone dial tone",
    "phone busy signal"
]

# --- Standalone Temporal NMS Function (Previously Static Method) ---
def apply_temporal_nms(
    events: List[Dict],
    min_gap_seconds: float = 1.0
) -> List[Dict]:
    """Apply temporal non-maximum suppression to events."""
    if not events:
        logging.debug("[NMS] Input events list is empty.") # Changed to DEBUG
        return []
    
    # Log input events
    logging.info(f"[NMS] Applying NMS to {len(events)} events with min_gap={min_gap_seconds}s") # Keep INFO for summary
    
    # Sort by confidence
    sorted_events = sorted(events, key=lambda x: x.get('confidence', 0.0), reverse=True)
    
    # Keep track of used time windows
    used_times = set()
    filtered_events = []
    
    logging.debug("[NMS] Processing sorted events:") # Changed to DEBUG
    for i, event in enumerate(sorted_events):
        start_time = event.get('start')
        confidence = event.get('confidence', 0.0)
        event_type = event.get('type', 'Unknown')
        
        # Ensure start time is valid
        if start_time is None or not isinstance(start_time, (int, float)):
            logging.warning(f"  [NMS Skip {i}] Invalid or missing start time for event: {event}")
            continue
            
        time_window = int(start_time)
        logging.debug(f"  [NMS Check {i}] Event: {event_type}@{start_time:.2f} (Conf: {confidence:.3f}, Window: {time_window})") # Changed to DEBUG
        
        # Check if any nearby windows are used
        nearby_used = False
        for t in range(
            time_window - int(min_gap_seconds) + 1,
            time_window + int(min_gap_seconds)
        ):
            if t in used_times:
                nearby_used = True
                logging.debug(f"    -> Nearby window {t} already used. Suppressing.") # Changed to DEBUG
                break
        
        if not nearby_used:
            logging.debug(f"    -> Keeping event. Marking window {time_window} as used.") # Changed to DEBUG
            filtered_events.append(event)
            used_times.add(time_window)
    
    # Sort by time for final output
    final_sorted_events = sorted(filtered_events, key=lambda x: x.get('start', 0.0))
    logging.info(f"[NMS] Filtered events count: {len(final_sorted_events)}") # Keep INFO for summary
    return final_sorted_events

# --- New Refactored Event Detection Function ---
def run_clap_event_detection(
    audio_data: np.ndarray,
    sample_rate: int,
    clap_model: ClapModel,
    clap_processor: ClapProcessor,
    device: torch.device,
    target_events: List[str],
    threshold: float = 0.98,
    chunk_duration: float = 5.0,
    min_gap: float = 1.0
) -> Dict[str, List[Dict]]:
    """
    Detects sound events using a pre-loaded CLAP model.

    Args:
        audio_data (np.ndarray): NumPy array containing audio waveform (float32).
        sample_rate (int): Sample rate of the audio data (must match CLAP_SAMPLE_RATE).
        clap_model (ClapModel): Pre-loaded CLAP model instance on the target device.
        clap_processor (ClapProcessor): Pre-loaded CLAP processor instance.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') the model is on.
        target_events (List[str]): List of event descriptions to detect.
        threshold (float): Confidence threshold for detection (0.0 to 1.0).
        chunk_duration (float): Duration of audio chunks to process (seconds).
        min_gap (float): Minimum gap between consecutive detections of the same event type after NMS (seconds).

    Returns:
        Dict[str, List[Dict]]: Dictionary mapping event types to lists of detected event dictionaries.
                                Each event dictionary contains 'start', 'end', 'type', 'confidence'.
    """
    if not target_events:
        logging.warning("No target events provided for CLAP detection.")
        return {}

    # Assert sample rate (should be done upstream, but double-check)
    if sample_rate != CLAP_SAMPLE_RATE:
         logging.error(f"CLAP Event Detection requires {CLAP_SAMPLE_RATE}Hz audio, received {sample_rate}Hz. Aborting.")
         # Or raise ValueError? For now, return empty.
         return {event: [] for event in target_events}

    logging.info("Starting CLAP event detection...")
    all_detected_events: Dict[str, List[Dict]] = {event: [] for event in target_events}

    try:
        # Precompute Text Features
        logging.info("Precomputing text features for event detection...")
        t_text_start = time.time()
        try:
             with torch.amp.autocast(device_type=device.type): # Use device.type for autocast
                text_inputs = clap_processor(
                    text=target_events,
                    return_tensors="pt",
                    padding=True
                )
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    precomputed_text_features = clap_model.get_text_features(**text_inputs)
             logging.info(f"Text features precomputed in {time.time() - t_text_start:.2f}s")
        except Exception as e:
             logging.error(f"Failed to precompute text features: {e}", exc_info=True)
             return all_detected_events # Return empty dict if text features fail

        # Process audio in chunks
        chunk_size = int(chunk_duration * sample_rate)
        num_chunks = int(np.ceil(len(audio_data) / chunk_size))
        
        logging.info(f"Processing audio in {num_chunks} chunks...")

        with tqdm(total=num_chunks, desc="Detecting events") as pbar:
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(audio_data))
                chunk = audio_data[start_idx:end_idx]

                # Skip very short final chunks if necessary
                if len(chunk) < sample_rate * 0.1: # Less than 100ms
                    logging.debug(f"Skipping very short final chunk {i}") # DEBUG level
                    continue

                # --- Process Audio Chunk ---
                try:
                    logging.debug(f"[CHUNKPROC {i}] Processing audio...") # DEBUG level
                    t_proc_start = time.time()
                    with torch.amp.autocast(device_type=device.type): # Use device.type
                        logging.debug(f"[CHUNKPROC {i}] Entering autocast context: {time.time() - t_proc_start:.4f}s") # DEBUG level
                        t_inner_start = time.time()
                        
                        t_call_processor = time.time()
                        inputs = clap_processor(
                            audios=[chunk], # Processor takes single array or list -> Explicitly wrap chunk in list
                            sampling_rate=sample_rate,
                            return_tensors="pt",
                            padding=True
                        )
                        logging.debug(f"[CHUNKPROC {i}] self.processor() call took: {time.time() - t_call_processor:.4f}s") # DEBUG level
                        
                        t_call_to_device = time.time()
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        logging.debug(f"[CHUNKPROC {i}] inputs.to(device) call took: {time.time() - t_call_to_device:.4f}s") # DEBUG level
                        logging.debug(f"[CHUNKPROC {i}] Audio inputs processed and moved to device.") # DEBUG level
                        
                        # Get embeddings
                        logging.debug(f"[CHUNKPROC {i}] Getting audio features...") # DEBUG level
                        with torch.no_grad():
                            logging.debug(f"[CHUNKPROC {i}] Entering no_grad context: {time.time() - t_inner_start:.4f}s") # DEBUG level
                            logging.debug(f"[CHUNKPROC {i}] Model device: {next(clap_model.parameters()).device}") # DEBUG level
                            logging.debug(f"[CHUNKPROC {i}] Audio input device: {inputs['input_features'].device if 'input_features' in inputs else 'N/A'}") # DEBUG level
                            
                            t_call_get_features = time.time()
                            audio_features = clap_model.get_audio_features(**inputs)
                            logging.debug(f"[CHUNKPROC {i}] model.get_audio_features() call took: {time.time() - t_call_get_features:.4f}s") # DEBUG level
                            logging.debug(f"[CHUNKPROC {i}] Audio features obtained.") # DEBUG level

                            # Calculate similarity scores using precomputed text_features
                            logging.debug(f"[CHUNKPROC {i}] Calculating similarity...") # DEBUG level
                            t_call_similarity = time.time()
                            similarity = torch.nn.functional.cosine_similarity(
                                audio_features[:, None],
                                precomputed_text_features[None, :], # Use precomputed
                                dim=-1
                            )
                            logging.debug(f"[CHUNKPROC {i}] similarity calculation took: {time.time() - t_call_similarity:.4f}s") # DEBUG level
                            logging.debug(f"[CHUNKPROC {i}] Similarity calculated.") # DEBUG level
                
                    # Find matches above threshold
                    matches = torch.where(similarity > threshold)
                    chunk_events = []
                    
                    for audio_idx, text_idx in zip(*matches):
                         # Map audio index within chunk back to overall time
                         # This simple mapping assumes one output feature vector per input audio
                         # which might not be precise for all models/layers, but is typical for CLAP.
                         event_time_in_chunk = (audio_idx / audio_features.shape[0]) * chunk_duration if audio_features.shape[0] > 0 else 0
                         
                         chunk_start_time = start_idx / sample_rate
                         event_start_time = chunk_start_time # Detection often applies to the whole chunk
                         event_end_time = chunk_start_time + (end_idx - start_idx) / sample_rate

                         chunk_events.append({
                             'type': target_events[text_idx],
                             'confidence': float(similarity[audio_idx, text_idx]),
                             'start': event_start_time, # Use chunk boundaries for now
                             'end': event_end_time    # Use chunk boundaries for now
                         })
                    
                    logging.debug(f"[CHUNKPROC {i}] Found {len(chunk_events)} potential events in chunk.") # DEBUG level

                    # Add chunk events to the main dictionary
                    for event in chunk_events:
                         all_detected_events[event['type']].append(event)

                except Exception as e:
                    logging.error(f"Error processing chunk {i}: {str(e)}")
                    logging.error("Full traceback:", exc_info=True)
                    # Continue to next chunk
                finally:
                    # Explicitly delete intermediate tensors
                    del inputs
                    del audio_features
                    del similarity
                    if device.type == 'cuda':
                         torch.cuda.empty_cache()
                    logging.debug(f"[CHUNKPROC {i}] Chunk processing finished, cache cleared.") # DEBUG level

                pbar.update(1)
                pbar.set_postfix({'Events': sum(len(e) for e in all_detected_events.values())})

        # Apply temporal NMS to each event type
        logging.info("Applying Temporal NMS...")
        final_events: Dict[str, List[Dict]] = {}
        for event_type, events in all_detected_events.items():
            final_events[event_type] = apply_temporal_nms(
                events,
                min_gap_seconds=min_gap
            )
            logging.info(f"  -> Found {len(final_events[event_type])} '{event_type}' events after NMS.")

        logging.info("CLAP event detection finished.")
        return final_events

    except Exception as e:
        logging.error(f"Error during CLAP event detection: {str(e)}")
        logging.error("Full traceback:", exc_info=True)
        return all_detected_events # Return whatever was collected before the error

# --- Remove Old detect_and_save_events function ---
# (This functionality will now be handled directly in whisperBite.py)

# --- Remove EventDetector Class ---
# (All logic is now in run_clap_event_detection or standalone NMS) 