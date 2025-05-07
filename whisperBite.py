import os
import sys
import argparse
import datetime
import logging
import json
import subprocess
import traceback
import re
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import threading
import time
import tempfile

# Third-party imports
import torch
import whisper
from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment
from pydub import AudioSegment
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, ClapModel, ClapProcessor
from yaml import SafeDumper
from sentence_transformers import SentenceTransformer, util

# Local imports
from utils import sanitize_filename, download_audio, zip_results, get_media_info, get_audio_channels
from vocal_separation import separate_vocals_with_demucs
from sound_detection import (
    extract_soundbites,
    TARGET_SOUND_PROMPTS,
    DEFAULT_CALL_CHUNK_DURATION,
    DEFAULT_CALL_THRESHOLD,
    CLAP_SAMPLE_RATE,
    detect_speech_regions_vad,
    extract_vad_aligned_soundbites
)
from event_detection import run_clap_event_detection, DEFAULT_EVENTS, CLAP_SAMPLE_RATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
# Module-level logger
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv')

# --- CLAP Model/Processor Loading (restored robust logic) ---
def load_clap_model_and_processor(model_id, device):
    import logging
    from transformers import ClapModel, ClapProcessor
    try:
        logging.info(f"Loading CLAP model ({model_id}) on device {device}...")
        model = ClapModel.from_pretrained(model_id).to(device)
        processor = ClapProcessor.from_pretrained(model_id)
        model.eval()
        logging.info("CLAP model and processor loaded successfully.")
        return model, processor
    except Exception as e:
        logging.error(f"Failed to load CLAP model/processor: {e}", exc_info=True)
        return None, None

def format_speaker_label(label: Union[str, int]) -> str:
    """Formats pyannote speaker labels (e.g., SPEAKER_00, 1) to S0, S1 etc."""
    if isinstance(label, int):
        return f"S{label}"
    elif isinstance(label, str):
        # Extract number, handling potential "SPEAKER_" prefix
        parts = label.split('_')
        num_part = parts[-1]
        try:
            num = int(num_part)
            return f"S{num}"
        except ValueError:
            logger.warning(f"Could not format speaker label '{label}', using original.")
            return label # Fallback to original label if parsing fails
    else:
        logger.warning(f"Unexpected speaker label type: {type(label)}. Using as is.")
        return str(label) # Fallback for unexpected types

def normalize_audio(input_audio, output_dir, target_lufs=-16):
    """Normalize audio to target LUFS using ffmpeg."""
    normalized_dir = os.path.join(output_dir, "normalized")
    os.makedirs(normalized_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    output_file = os.path.join(normalized_dir, f"{base_name}_normalized.wav")
    
    # Use ffmpeg for consistent normalization
    cmd = [
        "ffmpeg", "-y", "-i", input_audio,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", "48000", "-acodec", "pcm_s16le", output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logging.info(f"Normalized audio saved to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logging.error(f"Error normalizing audio: {e}")
        logging.error(f"STDERR: {e.stderr.decode('utf-8')}")
        raise

def detect_optimal_speakers(diarization_pipeline, audio_file, min_speakers=1, max_speakers=10):
    """Find optimal speaker count using clustering quality metrics."""
    best_score = -float('inf')
    best_num_speakers = 2  # Default fallback
    
    # Try different speaker counts and use silhouette score to evaluate
    for num_speakers in range(min_speakers, min(max_speakers + 1, 6)):  # Limit to reasonable range
        try:
            diarization = diarization_pipeline(audio_file, num_speakers=num_speakers)
            
            # Extract speaker segments for clustering quality assessment
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.end - turn.start,
                    'speaker': speaker
                })
            
            # Simplistic quality metric: longer segments are better (less fragmentation)
            avg_segment_duration = sum(s['duration'] for s in segments) / len(segments) if segments else 0
            num_speaker_changes = len(segments) - 1
            
            # Penalize excessive fragmentation while promoting longer segments
            score = avg_segment_duration * 10 - num_speaker_changes * 0.1
            
            logging.info(f"Speaker count {num_speakers}: score={score:.2f} (avg_duration={avg_segment_duration:.2f}s, changes={num_speaker_changes})")
            
            if score > best_score:
                best_score = score
                best_num_speakers = num_speakers
        
        except Exception as e:
            logging.warning(f"Error evaluating {num_speakers} speakers: {e}")
            continue
    
    logging.info(f"Selected optimal speaker count: {best_num_speakers}")
    return best_num_speakers

def merge_diarization_turns(diarization, max_silence_gap: float = 1.0, min_block_duration: float = 0.5) -> List[Dict[str, Any]]:
    """
    Merges consecutive speaker turns from the SAME speaker in Pyannote output
    if the gap between them is less than max_silence_gap.

    Args:
        diarization: The output Timeline from pyannote.audio Pipeline.
        max_silence_gap (float): Maximum silence duration (seconds) allowed
                                 between turns of the same speaker to merge.
        min_block_duration (float): Minimum duration (seconds) for a merged
                                    block to be kept.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents
                    a merged speaker block with keys:
                    'speaker', 'block_start', 'block_end', 'duration',
                    'turns' (list of original pyannote Segments within the block).
    """
    merged_blocks = []
    if not diarization:
        return []

    # Get turns sorted by start time (pyannote Timeline is usually sorted)
    turns = list(diarization.itertracks(yield_label=True))
    if not turns:
        return []

    # Sort by start time just in case it's not guaranteed
    turns.sort(key=lambda x: x[0].start)

    current_block_speaker = None
    current_block_start = None
    current_block_end = None
    current_block_turns = []

    for turn_segment, _, speaker_label_orig in turns:
        speaker_label = format_speaker_label(speaker_label_orig)

        if current_block_speaker is None:
            # Start the first block
            current_block_speaker = speaker_label
            current_block_start = turn_segment.start
            current_block_end = turn_segment.end
            current_block_turns = [turn_segment]
        else:
            # Check if the speaker is the same and the gap is small enough
            gap = turn_segment.start - current_block_end
            if speaker_label == current_block_speaker and gap <= max_silence_gap:
                # Merge: Extend the current block's end time and add the turn
                current_block_end = max(current_block_end, turn_segment.end)
                current_block_turns.append(turn_segment)
            else:
                # Speaker changed OR gap is too large, finish the previous block
                duration = current_block_end - current_block_start
                if duration >= min_block_duration:
                     merged_blocks.append({
                        'speaker': current_block_speaker,
                        'block_start': current_block_start,
                        'block_end': current_block_end,
                        'duration': duration,
                        'turns': current_block_turns # Store original Segment objects
                    })

                # Start a new block with the current turn
                current_block_speaker = speaker_label
                current_block_start = turn_segment.start
                current_block_end = turn_segment.end
                current_block_turns = [turn_segment]

    # Add the last block if it meets the minimum duration
    if current_block_speaker is not None:
        duration = current_block_end - current_block_start
        if duration >= min_block_duration:
             merged_blocks.append({
                'speaker': current_block_speaker,
                'block_start': current_block_start,
                'block_end': current_block_end,
                'duration': duration,
                'turns': current_block_turns
            })

    logger.info(f"Merged {len(turns)} speaker turns into {len(merged_blocks)} speaker blocks (max_gap={max_silence_gap}s, min_duration={min_block_duration}s).")
    return merged_blocks

def extract_audio_segment(
    source_audio_path: str,
    start_time: float,
    end_time: float,
    output_path: Optional[str] = None
) -> Union[str, Tuple[np.ndarray, int]]:
    """
    Extracts an audio segment from a source file using start and end times.

    Args:
        source_audio_path (str): Path to the source audio file.
        start_time (float): Start time of the segment in seconds.
        end_time (float): End time of the segment in seconds.
        output_path (Optional[str]): If provided, saves the segment to this path.
                                     Otherwise, returns the audio data and sample rate.

    Returns:
        Union[str, Tuple[np.ndarray, int]]: Path to the saved segment file if output_path is given,
                                             otherwise a tuple (audio_data, sample_rate).

    Raises:
        FileNotFoundError: If the source audio file doesn't exist.
        ValueError: If start_time or end_time are invalid.
        Exception: For other audio processing errors.
    """
    if not os.path.exists(source_audio_path):
        raise FileNotFoundError(f"Source audio file not found: {source_audio_path}")

    try:
        with sf.SoundFile(source_audio_path, 'r') as infile:
            sample_rate = infile.samplerate
            channels = infile.channels
            if start_time < 0 or end_time < 0 or start_time >= end_time or end_time > infile.frames / sample_rate:
                 # Allow end_time to be slightly beyond duration due to float precision
                 if end_time > (infile.frames / sample_rate) + 0.01:
                    raise ValueError(f"Invalid start/end times ({start_time:.2f}s / {end_time:.2f}s) for file duration {infile.frames / sample_rate:.2f}s")
                 end_time = min(end_time, infile.frames / sample_rate) # Clip end time precisely
                 if start_time >= end_time:
                     raise ValueError(f"Start time ({start_time:.2f}s) cannot be >= end time ({end_time:.2f}s) after clipping.")


            start_frame = int(start_time * sample_rate)
            # Calculate frames to read, ensuring it doesn't exceed file boundaries
            frames_to_read = min(int((end_time - start_time) * sample_rate), infile.frames - start_frame)

            if frames_to_read <= 0:
                 logger.warning(f"Calculated zero frames to read for segment {start_time:.2f}s - {end_time:.2f}s. Skipping extraction.")
                 # Return empty data or handle as needed
                 if output_path:
                     # Create an empty file? Or raise error? Let's return None for path
                     return None # Indicate failure to extract non-empty segment
                 else:
                     return (np.array([]), sample_rate) # Return empty numpy array


            infile.seek(start_frame)
            audio_data = infile.read(frames=frames_to_read, dtype='float32') # Read as float32 for Whisper

            if output_path:
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Write using soundfile
                sf.write(output_path, audio_data, sample_rate, format='WAV', subtype='PCM_16') # Save as 16-bit PCM WAV
                logger.debug(f"Extracted segment ({start_time:.2f}s - {end_time:.2f}s) to {output_path}")
                return output_path
            else:
                logger.debug(f"Extracted segment ({start_time:.2f}s - {end_time:.2f}s) to memory.")
                return (audio_data, sample_rate)

    except Exception as e:
        logger.error(f"Error extracting audio segment ({start_time:.2f}s - {end_time:.2f}s) from {source_audio_path}: {e}")
        raise # Re-raise the exception

# --- Workflow Parser ---
def parse_workflow_yaml(yaml_path: str) -> List[dict]:
    """
    Parse a workflow YAML file and return an ordered list of steps with parameters.
    Each step is a dict: {"name": step_name, "params": {...}}
    """
    with open(yaml_path, 'r') as f:
        workflow_def = yaml.safe_load(f)
    return workflow_def.get('steps', [])

# --- Modular Step Dispatcher ---
PIPELINE_STEPS = {}

def register_step(name: str):
    def decorator(func: Callable):
        PIPELINE_STEPS[name] = func
        return func
    return decorator

# Add a diagnostic mode flag (can be set via param or env var)
import os as _os
DIAGNOSTIC_MODE = bool(int(_os.environ.get('WB_DIAGNOSTIC_MODE', '0')))

def log_file_info(label, path):
    if not path or not os.path.exists(path):
        logger.error(f"[{label}] File does not exist: {path}")
        return False
    try:
        info = sf.info(path)
        logger.info(f"[{label}] File: {path}, Duration: {info.duration:.2f}s, Samplerate: {info.samplerate}, Channels: {info.channels}")
        if info.duration < 0.1:
            logger.warning(f"[{label}] File is very short (<0.1s): {path}")
        return True
    except Exception as e:
        logger.error(f"[{label}] Could not get info for {path}: {e}")
        return False

def log_context(context, step_name):
    logger.info(f"[CTX] Before {step_name}: keys={list(context.keys())}")
    for k in context:
        if 'path' in k or k.endswith('_path'):
            logger.info(f"[CTX] {k}: {context[k]}")

# --- Example: Add to each step ---
@register_step('normalize')
def step_normalize(context, params):
    input_file = context['input_file']
    output_dir = context['output_dir']
    if DIAGNOSTIC_MODE:
        log_file_info('normalize-input', input_file)
        log_context(context, 'normalize')
    normalized_audio_path = normalize_audio(input_file, output_dir, **params)
    if DIAGNOSTIC_MODE:
        log_file_info('normalize-output', normalized_audio_path)
    context['normalized_audio_path'] = normalized_audio_path
    return context

@register_step('demucs')
def step_demucs(context, params):
    normalized_audio_path = context.get('normalized_audio_path')
    output_dir = context['output_dir']
    if DIAGNOSTIC_MODE:
        log_file_info('demucs-input', normalized_audio_path)
        log_context(context, 'demucs')
    model = params.get('model', 'htdemucs')
    demucs_output_dir = os.path.join(output_dir, 'demucs')
    os.makedirs(demucs_output_dir, exist_ok=True)
    vocals_path, no_vocals_path = separate_vocals_with_demucs(
        normalized_audio_path,
        output_dir=demucs_output_dir,
        model=model
    )
    if DIAGNOSTIC_MODE:
        log_file_info('demucs-vocals', vocals_path)
        log_file_info('demucs-nonvocals', no_vocals_path)
    context['demucs_vocals_path'] = vocals_path
    context['demucs_nonvocals_path'] = no_vocals_path
    return context

@register_step('clap')
def step_clap(context, params):
    import soundfile as sf
    import numpy as np
    import logging
    # Use the fused model for general use (recommended by LAION)
    model_id = params.get('model_id', 'laion/clap-htsat-fused')
    device = context.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, str):
        device = torch.device(device)
    clap_model, clap_processor = load_clap_model_and_processor(model_id, device)
    if not clap_model or not clap_processor:
        logger.error("CLAP model/processor not loaded. Aborting CLAP step.")
        context['clap_segments'] = []
        return context
    audio_path = params.get('audio_path', context.get('normalized_audio_path'))
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"CLAP audio file missing: {audio_path}")
        context['clap_segments'] = []
        return context
    # Load audio, ensure 48kHz mono
    audio_data, sr = sf.read(audio_path, dtype='float32')
    logger.info(f"[CLAP] Audio path: {audio_path}, SR: {sr}, Shape: {audio_data.shape}")
    if sr != 48000:
        logger.warning(f"Resampling audio from {sr}Hz to 48000Hz for CLAP.")
        # ... (resample logic here, or fallback)
    if audio_data.ndim > 1:
        logger.info(f"Flattening audio to mono. Shape: {audio_data.shape}")
        audio_data = audio_data[:, 0]
    # Parameters
    threshold = float(params.get('threshold', 0.5))
    chunk_duration = float(params.get('chunk_duration_s', 5.0))
    min_gap = float(params.get('min_gap', 1.0))
    prompts = params.get('prompts', ["speech", "music", "telephone ringing", "applause"])  # sensible defaults
    logger.info(f"[CLAP] Threshold: {threshold}, Chunk: {chunk_duration}s, Min gap: {min_gap}s, Prompts: {prompts}")
    # Run event detection
    from event_detection import run_clap_event_detection
    detected_events = run_clap_event_detection(
        audio_data=audio_data,
        sample_rate=48000,
        clap_model=clap_model,
        clap_processor=clap_processor,
        device=device,
        target_events=prompts,
        threshold=threshold,
        chunk_duration=chunk_duration,
        min_gap=min_gap
    )
    # Flatten events to segments
    segments = []
    for event_type, events in detected_events.items():
        for e in events:
            segments.append({
                'start': e['start'],
                'end': e['end'],
                'type': event_type,
                'confidence': e['confidence']
            })
    segments.sort(key=lambda x: x['start'])
    logger.info(f"[CLAP] Detected {len(segments)} segments.")
    context['clap_segments'] = segments
    context['clap_model_loaded'] = True
    context['clap_params'] = dict(threshold=threshold, chunk_duration=chunk_duration, min_gap=min_gap, prompts=prompts)
    # If no segments, fallback to VAD/Whisper
    if not segments:
        logger.warning("CLAP detected no events, falling back to VAD/Whisper.")
        context['clap_segments'] = []
        context['clap_model_loaded'] = False
    return context

@register_step('diarization')
def step_diarization(context, params):
    import torch
    import os
    from pyannote.audio import Pipeline
    import gc
    hf_token = context['config'].get('hf_token')
    device = context.get('device', 'cpu')
    if isinstance(device, str):
        device = torch.device(device)
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(device)
    diarization_params = {}
    if params.get('num_speakers') and not params.get('auto_speakers', False):
        diarization_params['num_speakers'] = params['num_speakers']
    # Use VAD/CLAP chunks for diarization
    vad_chunks = context.get('vad_chunks', [])
    vocals_path = context.get('demucs_vocals_path')
    normalized_path = context.get('normalized_audio_path')
    diarization_audio = vocals_path if vocals_path and os.path.exists(vocals_path) else normalized_path
    logger.info(f"[DIARIZATION] Using audio for diarization: {diarization_audio}")
    diarization_result = []
    if vad_chunks:
        logger.info(f"[DIARIZATION] Running chunked diarization on {len(vad_chunks)} chunks.")
        for idx, chunk in enumerate(vad_chunks):
            start = chunk['start']
            end = chunk['end']
            logger.info(f"[DIARIZATION] Processing chunk {idx+1}/{len(vad_chunks)}: {start:.2f}-{end:.2f}s")
            # Extract chunk audio to temp file
            import tempfile
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmpf:
                try:
                    # Read and write the chunk
                    with sf.SoundFile(diarization_audio, 'r') as infile:
                        sr = infile.samplerate
                        infile.seek(int(start * sr))
                        frames = int((end - start) * sr)
                        audio_data = infile.read(frames)
                        sf.write(tmpf.name, audio_data, sr)
                    # Run diarization on the chunk
                    if device.type == 'cuda':
                        vram_alloc = torch.cuda.memory_allocated(device) / 1024**2
                        vram_reserved = torch.cuda.memory_reserved(device) / 1024**2
                        logger.info(f"[VRAM] Before chunk {idx+1}: Allocated={vram_alloc:.1f}MB, Reserved={vram_reserved:.1f}MB")
                    chunk_timeline = diarization_pipeline(tmpf.name, **diarization_params)
                    for turn, _, label in chunk_timeline.itertracks(yield_label=True):
                        diarization_result.append({
                            "speaker": format_speaker_label(label),
                            "start": turn.start + start,
                            "end": turn.end + start
                        })
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                        vram_alloc = torch.cuda.memory_allocated(device) / 1024**2
                        vram_reserved = torch.cuda.memory_reserved(device) / 1024**2
                        logger.info(f"[VRAM] After chunk {idx+1}: Allocated={vram_alloc:.1f}MB, Reserved={vram_reserved:.1f}MB")
                except Exception as e:
                    logger.error(f"[DIARIZATION] Failed on chunk {idx+1}: {e}")
    else:
        logger.info("[DIARIZATION] No vad_chunks found, running diarization on full file.")
        diarization_timeline = diarization_pipeline(diarization_audio, **diarization_params)
        diarization_result = [
            {"speaker": format_speaker_label(label), "start": turn.start, "end": turn.end}
            for turn, _, label in diarization_timeline.itertracks(yield_label=True)
        ]
    context['diarization'] = diarization_result
    return context

@register_step('vad')
def step_vad(context, params):
    audio_path = params.get('audio_path', context.get('demucs_vocals_path', context.get('normalized_audio_path')))
    threshold = params.get('threshold', 0.5)
    vad_regions = detect_speech_regions_vad(audio_path, threshold=threshold)
    context['vad_regions'] = vad_regions
    return context

@register_step('whisper')
def step_whisper(context, params):
    import whisper
    audio_path = params.get('audio_path', context.get('normalized_audio_path'))
    if DIAGNOSTIC_MODE:
        log_file_info('whisper-input', audio_path)
        log_context(context, 'whisper')
    model_size = params.get('model_size', 'medium')
    device = context.get('device', 'cpu')
    whisper_model = whisper.load_model(model_size, device=device)
    conversation_blocks = context.get('conversation_blocks')
    if conversation_blocks:
        logger.info(f"[WHISPER] Transcribing {len(conversation_blocks)} segments...")
        for block in conversation_blocks:
            seg_audio = block['segment_audio_path']
            result = whisper_model.transcribe(seg_audio)
            block['transcription'] = result.get('text', '')
            block['word_timestamps'] = result.get('segments', [])
            logger.info(f"[WHISPER] Block {block.get('block_id', 'call')} transcript: {block['transcription'][:60]}...")
        logger.info(f"[WHISPER] Finished transcription. Blocks: {len(conversation_blocks)}. Keys: {list(conversation_blocks[0].keys()) if conversation_blocks else 'N/A'}")
        context['conversation_blocks'] = conversation_blocks
        context['results']['conversation_blocks'] = conversation_blocks
    else:
        result = whisper_model.transcribe(audio_path)
        logger.info(f"[WHISPER] Transcription result: {result}")
        context['transcription'] = result
    return context

@register_step('annotate_segments')
def step_annotate_segments(context, params):
    conversation_blocks = context.get('conversation_blocks', [])
    clap_events = context.get('clap_segments', [])
    for block in conversation_blocks:
        block_start = block.get('start', 0)
        block_end = block.get('end', 0)
        block['contextual_events'] = []
        for event in clap_events:
            event_start = event.get('start', 0)
            event_end = event.get('end', 0)
            if max(block_start, event_start) < min(block_end, event_end):
                block['contextual_events'].append(event)
    context['conversation_blocks'] = conversation_blocks
    context['results']['conversation_blocks'] = conversation_blocks
    return context

@register_step('soundbite')
def step_soundbite(context, params):
    if not context.get('speaker_soundbites'):
        logger.error("[SOUNDBITE] No speaker_soundbites found. No soundbites will be written.")
        context['soundbite_metadata'] = []
        context['soundbite_error'] = 'No soundbites were produced. Upstream diarization or transcription failed.'
        return context
    import os
    import yaml
    from utils import sanitize_filename
    output_dir = context.get('output_dir', 'soundbites')
    speakers_dir = os.path.join(output_dir, 'speakers')
    os.makedirs(speakers_dir, exist_ok=True)
    speaker_soundbites = context.get('speaker_soundbites', [])
    clap_events = context.get('clap_segments', []) or context.get('clap_events', [])
    soundbite_metadata = []
    sequence_counter = 1
    for bite in speaker_soundbites:
        speaker = bite.get('speaker', 'UNK')
        transcript = bite.get('transcription', '').strip()
        start = bite.get('start_time', 0)
        end = bite.get('end_time', 0)
        # Find overlapping CLAP events
        overlapping_events = []
        for event in clap_events:
            event_start = event.get('start', event.get('start_time', 0))
            event_end = event.get('end', event.get('end_time', 0))
            if max(start, event_start) < min(end, event_end):
                overlapping_events.append(event)
        # Append event markers to transcript text
        transcript_with_events = transcript
        for event in overlapping_events:
            event_type = event.get('type', 'event')
            event_time = event.get('start', event.get('start_time', 0))
            transcript_with_events += f" [EVENT: {event_type} at {event_time:.2f}s]"
        # Canonical filename: 0001_hello_this_is_bob
        snippet = sanitize_filename(transcript)[:128]
        seq_str = f"{sequence_counter:04d}"
        base_name = f"{seq_str}_{snippet}"
        speaker_dir = os.path.join(speakers_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        wav_path = os.path.join(speaker_dir, f"{base_name}.wav")
        txt_path = os.path.join(speaker_dir, f"{base_name}.txt")
        yaml_path = os.path.join(speaker_dir, f"{base_name}.yaml")
        # Copy/copy2 the audio file
        import shutil
        try:
            shutil.copy2(bite['soundbite_path'], wav_path)
        except Exception as e:
            logger.error(f"Failed to copy soundbite {bite['soundbite_path']} to {wav_path}: {e}")
            continue
        # Write transcript with event markers
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript_with_events)
        # Write YAML metadata with events
        metadata = {
            'sequence': sequence_counter,
            'speaker': speaker,
            'start': start,
            'end': end,
            'transcript': transcript,
            'transcript_with_events': transcript_with_events,
            'clap_events': overlapping_events,
            'call_id': bite.get('call_id', 0),
            'file_paths': {
                'audio': os.path.relpath(wav_path, output_dir),
                'transcript': os.path.relpath(txt_path, output_dir),
                'yaml': os.path.relpath(yaml_path, output_dir)
            },
            'parent_call_audio': bite.get('parent_call_audio', ''),
            'original_input': bite.get('original_input', '')
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, allow_unicode=True, sort_keys=False)
        # Update bite with paths and event context
        bite['wav_path'] = wav_path
        bite['txt_path'] = txt_path
        bite['yaml_path'] = yaml_path
        bite['sequence'] = sequence_counter
        bite['clap_events'] = overlapping_events
        bite['transcript_with_events'] = transcript_with_events
        soundbite_metadata.append(metadata)
        sequence_counter += 1
    context['soundbite_metadata'] = soundbite_metadata
    context['speaker_soundbites'] = speaker_soundbites
    logger.info(f"[SOUNDBITE] Wrote {sequence_counter-1} canonical soundbites with metadata and event context.")
    return context

@register_step('write_transcripts')
def step_write_transcripts(context, params):
    if not context.get('speaker_soundbites'):
        logger.error("[WRITE_TRANSCRIPTS] No speaker_soundbites found. No transcripts will be written.")
        context['write_transcripts_error'] = 'No transcripts were produced. Upstream soundbite step failed.'
        return context
    # Write per-segment/per-speaker TXT files in canonical structure (with event markers)
    speaker_soundbites = context.get('speaker_soundbites', [])
    for bite in speaker_soundbites:
        txt_path = bite.get('txt_path')
        transcript_with_events = bite.get('transcript_with_events', bite.get('transcription', '').strip())
        if txt_path and transcript_with_events:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(transcript_with_events)
    logger.info(f"[TRANSCRIPT] Ensured all per-segment/per-speaker TXT files are written with event markers.")
    return context

def warn_if_clap_segment_used():
    logger.warning("[CLAP_SEGMENT] CLAP-based segmentation is disabled. CLAP is only used for annotation/context after all segmentation and transcription.")

# Overwrite/disable the clap_segment step if called
@register_step('clap_segment')
def step_clap_segment(context, params):
    warn_if_clap_segment_used()
    return context

@register_step('diarize_and_split_speakers')
def step_diarize_and_split_speakers(context, params):
    from pyannote.audio import Pipeline
    from utils import extract_audio_segment, sanitize_filename
    import torch
    import os
    import soundfile as sf
    import gc
    from whisper import load_model
    import threading
    import time
    import logging
    import numpy as np
    import shutil
    import tempfile
    import traceback
    hf_token = context['config'].get('hf_token')
    device = context.get('device', 'cpu')
    if isinstance(device, str):
        device = torch.device(device)
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(device)
    whisper_model = load_model(params.get('whisper_model', 'base'))
    conversation_blocks = context.get('conversation_blocks', [])
    logger.info(f"[DIARIZE_SPLIT] Received {len(conversation_blocks)} blocks for processing.")
    debug_max_blocks = params.get('debug_max_blocks', None)
    if debug_max_blocks is not None:
        conversation_blocks = conversation_blocks[:int(debug_max_blocks)]
        logger.info(f"[DIARIZE_SPLIT] Debug mode: only processing first {debug_max_blocks} blocks.")
    for i, block in enumerate(conversation_blocks[:5]):
        logger.info(f"[DIARIZE_SPLIT] Block {i}: {block}")
    speaker_soundbites = []
    seen_segments = set()  # To deduplicate by (speaker, start, end)
    output_dir = context.get('output_dir', '.')
    soundbites_dir = os.path.join(output_dir, 'speaker_soundbites')
    os.makedirs(soundbites_dir, exist_ok=True)
    num_written = 0
    for idx, block in enumerate(conversation_blocks):
        demucs_vocals_path = context.get('demucs_vocals_path')
        seg_audio = demucs_vocals_path if demucs_vocals_path and os.path.exists(demucs_vocals_path) else block['segment_audio_path']
        seg_start = block['start_time']
        seg_end = block['end_time']
        # --- Segment validity checks ---
        if not seg_audio or not os.path.exists(seg_audio):
            logger.error(f"[DIARIZE_SPLIT] Segment audio file does not exist: {seg_audio}. Skipping block {idx+1}.")
            continue
        try:
            with sf.SoundFile(seg_audio, 'r') as infile:
                duration = infile.frames / infile.samplerate
                file_size = os.path.getsize(seg_audio)
        except Exception as e:
            logger.error(f"[DIARIZE_SPLIT] Failed to read duration for {seg_audio}: {e}. Skipping block {idx+1}.")
            continue
        if duration < 0.1 or file_size == 0:
            logger.error(f"[DIARIZE_SPLIT] Segment {seg_audio} is too short or empty (duration={duration:.2f}s, size={file_size} bytes). Skipping block {idx+1}.")
            continue
        if device.type == 'cuda':
            vram_alloc = torch.cuda.memory_allocated(device) / 1024**2
            vram_reserved = torch.cuda.memory_reserved(device) / 1024**2
            logger.info(f"[VRAM] Before block {idx+1}/{len(conversation_blocks)}: Allocated={vram_alloc:.1f}MB, Reserved={vram_reserved:.1f}MB")
        # --- Timeout wrapper for diarization ---
        diarization_result = {'error': None, 'diarization': None}
        def diarize_with_catch():
            try:
                diarization_result['diarization'] = diarization_pipeline(seg_audio)
            except RuntimeError as oom:
                if 'out of memory' in str(oom).lower() and device.type == 'cuda':
                    logger.error(f"[DIARIZE_SPLIT] CUDA OOM on block {idx+1}/{len(conversation_blocks)}. Skipping this block. Consider using a smaller model or running on CPU.")
                    torch.cuda.empty_cache()
                else:
                    logger.error(f"[DIARIZE_SPLIT] RuntimeError on block {idx+1}: {oom}")
                diarization_result['error'] = str(oom)
            except Exception as e:
                logger.error(f"[DIARIZE_SPLIT] Error on block {idx+1}: {e}")
                diarization_result['error'] = str(e)
        diar_thread = threading.Thread(target=diarize_with_catch)
        diar_thread.start()
        diar_thread.join(timeout=60)  # 60 second timeout
        if diar_thread.is_alive():
            logger.error(f"[DIARIZE_SPLIT] Diarization timed out for block {idx+1}/{len(conversation_blocks)} (file: {seg_audio}). Skipping.")
            # Optionally, save the problematic segment for later inspection
            try:
                debug_dir = os.path.join(output_dir, 'debug_hangs')
                os.makedirs(debug_dir, exist_ok=True)
                shutil.copy2(seg_audio, os.path.join(debug_dir, f"block_{idx+1:04d}_timeout.wav"))
            except Exception as e:
                logger.warning(f"[DIARIZE_SPLIT] Failed to save debug copy of timed-out segment: {e}")
            continue
        if diarization_result['error'] or diarization_result['diarization'] is None:
            logger.error(f"[DIARIZE_SPLIT] Diarization failed for block {idx+1}/{len(conversation_blocks)} (file: {seg_audio}). Error: {diarization_result['error']}")
            continue
        diarization = diarization_result['diarization']
        for turn, _, label in diarization.itertracks(yield_label=True):
            speaker = format_speaker_label(label)
            start = turn.start
            end = turn.end
            if end > duration:
                logger.warning(f"[DIARIZE_SPLIT] Clipping end time {end:.2f}s to file duration {duration:.2f}s for {seg_audio}")
                end = duration
            if start >= end:
                logger.warning(f"[DIARIZE_SPLIT] Skipping segment: start {start:.2f}s >= end {end:.2f}s for {seg_audio}")
                continue
            seg_key = (speaker, round(start, 3), round(end, 3))
            if seg_key in seen_segments:
                logger.info(f"[DIARIZE_SPLIT] Skipping duplicate segment: {seg_key}")
                continue
            seen_segments.add(seg_key)
            soundbite_path = os.path.join(
                soundbites_dir,
                f"{block['block_id']}_{sanitize_filename(speaker)}_{start:.2f}_{end:.2f}.wav"
            )
            os.makedirs(os.path.dirname(soundbite_path), exist_ok=True)
            extract_audio_segment(seg_audio, start, end, soundbite_path)
            logger.info(f"[DIARIZE_SPLIT] Wrote audio file: {soundbite_path}")
            try:
                result = whisper_model.transcribe(soundbite_path)
                transcript = result.get('text', '')
                word_timestamps = result.get('segments', [])
            except Exception as e:
                logger.error(f"Failed to transcribe {soundbite_path}: {e}")
                transcript = ''
                word_timestamps = []
            speaker_soundbites.append({
                'parent_block_id': block['block_id'],
                'speaker': speaker,
                'start_time': start,
                'end_time': end,
                'soundbite_path': soundbite_path,
                'transcription': transcript,
                'word_timestamps': word_timestamps
            })
            num_written += 1
            logger.info(f"[DIARIZE_SPLIT] Wrote transcript for: {soundbite_path}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            vram_alloc = torch.cuda.memory_allocated(device) / 1024**2
            vram_reserved = torch.cuda.memory_reserved(device) / 1024**2
            logger.info(f"[VRAM] After block {idx+1}/{len(conversation_blocks)}: Allocated={vram_alloc:.1f}MB, Reserved={vram_reserved:.1f}MB")
    logger.info(f"[DIARIZE_SPLIT] Created {len(speaker_soundbites)} unique per-speaker soundbites. {num_written} files written.")
    if num_written == 0:
        logger.warning("[DIARIZE_SPLIT] No unique segments were processed. No output files were written. Check upstream chunking and deduplication.")
    context['speaker_soundbites'] = speaker_soundbites
    context['results']['speaker_soundbites'] = speaker_soundbites
    return context

# --- Insert this helper function ---
def process_segment_with_old_pipeline(segment_path, output_dir, model_name, enable_vocal_separation, num_speakers, auto_speakers, enable_word_extraction, enable_second_pass, attempt_sound_detection):
    """Run the 'evenolder' pipeline on a single audio segment, with Demucs always run after normalization."""
    from pyannote.audio import Pipeline
    import torch
    model = whisper.load_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization").to(device)
    # 1. Normalize
    normalized_file = normalize_audio(segment_path, output_dir)
    # 2. Always run Demucs
    vocals_file, no_vocals_file = separate_vocals_with_demucs(normalized_file, output_dir)
    pipeline_audio_input = vocals_file if vocals_file and os.path.exists(vocals_file) else normalized_file
    # 3. Transcribe the full call segment (vocals)
    call_transcript = model.transcribe(pipeline_audio_input)
    # 4. Diarization on vocals
    actual_num_speakers = num_speakers
    if auto_speakers:
        try:
            actual_num_speakers = detect_optimal_speakers(pipeline, pipeline_audio_input)
        except Exception as e:
            logger.warning(f"Auto speaker detection failed: {e}. Using provided value: {num_speakers}")
            actual_num_speakers = num_speakers
    diarization = pipeline(pipeline_audio_input, num_speakers=actual_num_speakers)
    # 5. Slice by speaker (vocals)
    speaker_output_dir = os.path.join(output_dir, "speakers")
    segment_info_dict = slice_audio_by_speaker(pipeline_audio_input, diarization, speaker_output_dir)
    # 6. Transcribe each speaker soundbite (vocals)
    speaker_segments = transcribe_with_whisper(model, segment_info_dict, output_dir)
    # 7. Optionally, transcribe no_vocals for sound events
    sound_event_segments = []
    if attempt_sound_detection and no_vocals_file and os.path.exists(no_vocals_file):
        try:
            no_vocals_transcription = model.transcribe(no_vocals_file, word_timestamps=True)
            import re
            bracket_pattern = re.compile(r"^\[\s*.*\s*\]$|\(.*\)|♪.*♪")
            if 'segments' in no_vocals_transcription:
                for segment in no_vocals_transcription['segments']:
                    text = segment['text'].strip()
                    if text and bracket_pattern.fullmatch(text):
                        sound_event_segments.append({
                            'speaker': 'SOUND',
                            'start': segment['start'],
                            'end': segment['end'],
                            'text': text,
                            'audio_file': None,
                            'transcript_file': None,
                            'sequence': -1
                        })
        except Exception as e:
            logger.error(f"Error during sound detection on {no_vocals_file}: {e}")
    # Aggregate outputs
    return {
        'call_transcript': call_transcript,
        'speaker_segments': speaker_segments,
        'sound_event_segments': sound_event_segments
    }

def slice_audio_by_speaker(file_path, diarization, speaker_output_dir, min_segment_duration=1.0):
    from pydub import AudioSegment
    import os
    import json
    audio = AudioSegment.from_file(file_path)
    os.makedirs(speaker_output_dir, exist_ok=True)
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start
        if duration < min_segment_duration:
            continue
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append({
            'start': turn.start,
            'end': turn.end,
            'duration': duration
        })
    speaker_files = {}
    segment_info = {}
    segment_counter = 0
    for speaker_raw, segments in speaker_segments.items():
        formatted_speaker = format_speaker_label(speaker_raw)
        speaker_dir = os.path.join(speaker_output_dir, formatted_speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        if formatted_speaker not in speaker_files:
            speaker_files[formatted_speaker] = []
            segment_info[formatted_speaker] = []
        segments.sort(key=lambda x: x['start'])
        merged_segments = []
        if segments:
            current = segments[0]
            for next_segment in segments[1:]:
                gap = next_segment['start'] - current['end']
                if gap < 0.5:
                    current['end'] = next_segment['end']
                    current['duration'] = current['end'] - current['start']
                else:
                    merged_segments.append(current)
                    current = next_segment
            merged_segments.append(current)
        for i, segment in enumerate(merged_segments):
            segment_audio = audio[segment['start'] * 1000:segment['end'] * 1000]
            fade_duration = min(100, segment_audio.duration_seconds * 1000 / 4)
            segment_audio = segment_audio.fade_in(int(fade_duration)).fade_out(int(fade_duration))
            segment_filename = f"{segment_counter:04d}_segment_{int(segment['start'])}_{int(segment['end'])}_{int(segment['duration'])}.wav"
            segment_path = os.path.join(speaker_dir, segment_filename)
            segment_audio.export(segment_path, format="wav")
            speaker_files[formatted_speaker].append(segment_path)
            segment_info[formatted_speaker].append({
                'path': segment_path,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'sequence': segment_counter
            })
            segment_counter += 1
    with open(os.path.join(speaker_output_dir, "segments.json"), 'w') as f:
        json.dump(segment_info, f, indent=2)
    return segment_info

def transcribe_with_whisper(model, segment_info_dict, output_dir, enable_word_extraction=False):
    import os
    import json
    from pydub import AudioSegment
    all_transcriptions = {}
    word_timings = {}
    word_counter = 0
    for formatted_speaker, segments in segment_info_dict.items():
        speaker_transcription_dir = os.path.join(output_dir, f"{formatted_speaker}_transcriptions")
        os.makedirs(speaker_transcription_dir, exist_ok=True)
        words_dir = None
        if enable_word_extraction:
            words_dir = os.path.join(output_dir, f"{formatted_speaker}_words")
            os.makedirs(words_dir, exist_ok=True)
        speaker_full_transcript = ""
        segment_transcriptions = []
        segments.sort(key=lambda x: x['sequence'])
        for segment_info in segments:
            segment_path = segment_info['path']
            start_time = segment_info['start']
            end_time = segment_info['end']
            duration = segment_info['duration']
            seq_num = segment_info['sequence']
            segment_name = os.path.basename(segment_path)
            try:
                if duration < 1:
                    continue
                transcription = model.transcribe(segment_path, word_timestamps=enable_word_extraction)
                text = transcription['text'].strip()
                if not text:
                    continue
                if enable_word_extraction and 'segments' in transcription and transcription['segments']:
                    for segment in transcription['segments']:
                        if 'words' in segment:
                            try:
                                audio = AudioSegment.from_file(segment_path)
                            except Exception as audio_load_err:
                                continue
                            for word_data in segment['words']:
                                word = word_data['word'].strip()
                                if not word:
                                    continue
                                word_start = word_data['start']
                                word_end = word_data['end']
                                word_duration = word_end - word_start
                                if word_duration < 0.1:
                                    continue
                                padding_ms = max(100, int(word_duration * 1000 * 0.4))
                                word_start_ms = max(0, int((word_start * 1000) - padding_ms))
                                word_end_ms = min(len(audio), int((word_end * 1000) + padding_ms))
                                word_audio = audio[word_start_ms:word_end_ms]
                                fade_ms = max(30, int(word_audio.duration_seconds * 1000 * 0.1))
                                word_audio = word_audio.fade_in(fade_ms).fade_out(fade_ms)
                                word_filename = f"{word_counter:04d}_{word.replace(' ', '_')}.wav"
                                word_path = os.path.join(words_dir, word_filename)
                                word_audio.export(word_path, format="wav")
                                if formatted_speaker not in word_timings:
                                    word_timings[formatted_speaker] = []
                                word_timings[formatted_speaker].append({
                                    'word': word,
                                    'file': word_path,
                                    'start': start_time + word_start,
                                    'end': start_time + word_end,
                                    'duration': word_duration,
                                    'sequence': word_counter
                                })
                                word_counter += 1
                first_words = text.split()[:5]
                base_name = f"{seq_num:04d}_{sanitize_filename('_'.join(first_words))}"
                transcription_file = os.path.join(speaker_transcription_dir, f"{base_name}.txt")
                audio_output_file = os.path.join(speaker_transcription_dir, f"{base_name}.wav")
                with open(transcription_file, "w", encoding='utf-8') as f:
                    timestamp_str = f"[{start_time:.2f}s - {end_time:.2f}s]"
                    f.write(f"{timestamp_str} {text}")
                import shutil
                shutil.copy2(segment_path, audio_output_file)
                speaker_full_transcript += f"{timestamp_str} {text}\n\n"
                segment_transcriptions.append({
                    'speaker': formatted_speaker,
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'audio_file': audio_output_file,
                    'transcript_file': transcription_file,
                    'sequence': seq_num
                })
            except Exception as e:
                continue
        if speaker_full_transcript:
            with open(os.path.join(output_dir, f"{formatted_speaker}_full_transcript.txt"), 'w') as f:
                f.write(f"=== SPEAKER {formatted_speaker} TRANSCRIPT ===\n\n")
                f.write(speaker_full_transcript)
        all_transcriptions[formatted_speaker] = segment_transcriptions
    all_segments = []
    for speaker, segments in all_transcriptions.items():
        all_segments.extend(segments)
    all_segments.sort(key=lambda x: x['start'])
    return all_segments

# --- Refactored process_audio ---
def process_audio(
    input_file: str,
    output_dir: str,
    preset_name: str,
    preset_config: Dict[str, Any],
    stop_event: Optional[threading.Event] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Modular, YAML-driven process_audio workflow.
    """
    start_time_total = time.time()
    logger.info(f"Starting processing for: {input_file}")

    if stop_event and stop_event.is_set():
        logger.warning("Processing cancelled before start.")
        return {"status": "cancelled", "error": "Cancelled before start"}

    # --- Config & Output Setup ---
    try:
        config = preset_config
        workflow_yaml = config.get('workflow_yaml')
        workflow_steps = []
        if workflow_yaml:
            workflow_steps = parse_workflow_yaml(workflow_yaml)
        else:
            workflow_steps = config.get('workflow_steps', [])
        if not workflow_steps:
            logger.error("No workflow steps defined in config or YAML.")
            return {"status": "error", "error": "No workflow steps defined."}

        os.makedirs(output_dir, exist_ok=True)
        log_file_path = os.path.join(output_dir, "processing.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Processing run logs will be saved to: {log_file_path}")
        logger.info(f"Using preset: {preset_name}")

        # Redact hf_token from config before logging
        def redact_token(cfg):
            if isinstance(cfg, dict):
                cfg = cfg.copy()
                if 'hf_token' in cfg:
                    cfg['hf_token'] = '***REDACTED***'
                for k, v in cfg.items():
                    cfg[k] = redact_token(v)
            elif isinstance(cfg, list):
                cfg = [redact_token(i) for i in cfg]
            return cfg

        redacted_config = redact_token(config)
        logger.info(f"Full configuration: {json.dumps(redacted_config, indent=2)}")
        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved run configuration to {config_path}")
    except Exception as e:
        logger.exception("Error during initial setup or config loading.")
        return {"status": "error", "error": f"Setup/Config Error: {e}"}

    # Determine processing device
    try:
        processing_device = torch.device(device)
        logger.info(f"Using processing device: {processing_device}")
    except Exception as e:
        logger.warning(f"Invalid device specified '{device}', falling back to auto-detect. Error: {e}")
        processing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using fallback processing device: {processing_device}")

    # --- Modular Pipeline Execution ---
    context = {
        'input_file': input_file,
        'output_dir': output_dir,
        'preset': preset_name,
        'device': str(processing_device),
        'config': config,
        'results': {},
    }
    results = context['results']
    try:
        if DIAGNOSTIC_MODE:
            logger.info(f"[DIAG] Workflow steps: {[step['name'] for step in workflow_steps]}")
            logger.info(f"[DIAG] Initial input file: {input_file}")
        for step in workflow_steps:
            step_name = step['name']
            params = step.get('params', {})
            if stop_event and stop_event.is_set():
                logger.warning(f"Processing cancelled before step: {step_name}")
                results['status'] = 'cancelled'
                results['error'] = f"Cancelled before step: {step_name}"
                break
            if step_name in PIPELINE_STEPS:
                logger.info(f"Running pipeline step: {step_name}")
                try:
                    context = PIPELINE_STEPS[step_name](context, params)
                except Exception as step_e:
                    logger.error(f"Error in step '{step_name}': {step_e}", exc_info=True)
                    context['step_error'] = str(step_e)
                if progress_callback:
                    progress_callback(step_name, 'done', None)
            else:
                logger.warning(f"Unknown pipeline step: {step_name}")
        # --- Ensure critical outputs are always written ---
        try:
            context = PIPELINE_STEPS['write_transcripts'](context, {})
        except Exception as wt_e:
            logger.error(f"Error in write_transcripts: {wt_e}", exc_info=True)
        try:
            context = PIPELINE_STEPS['aggregate_outputs'](context, {})
        except Exception as ao_e:
            logger.error(f"Error in aggregate_outputs: {ao_e}", exc_info=True)
        results['status'] = results.get('status', 'completed')
        results['processing_end_time'] = datetime.datetime.now().isoformat()
        # Save master transcript YAML
        master_yaml_path = os.path.join(output_dir, "master_transcript.yaml")
        class CustomDumper(SafeDumper):
            def represent_data(self, data):
                if isinstance(data, np.integer):
                    return self.represent_int(int(data))
                if isinstance(data, np.floating):
                    return self.represent_float(float(data))
                if isinstance(data, np.ndarray):
                    return self.represent_list(data.tolist())
                if isinstance(data, Segment):
                    return self.represent_dict({"start": data.start, "end": data.end})
                if isinstance(data, datetime.datetime):
                    return self.represent_scalar('tag:yaml.org,2002:timestamp', data.isoformat())
                return super().represent_data(data)
        CustomDumper.add_representer(Segment, lambda dumper, data: dumper.represent_dict({"start": data.start, "end": data.end}))
        CustomDumper.add_representer(datetime.datetime, lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:timestamp', data.isoformat()))
        with open(master_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, Dumper=CustomDumper, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"Saved master transcript to {master_yaml_path}")
    except Exception as e:
        logger.exception(f"An error occurred during modular pipeline processing: {e}")
        results['status'] = 'error'
        results['error'] = traceback.format_exc()
        if progress_callback:
            progress_callback("Overall Process", "error", results['error'])
    finally:
        if 'file_handler' in locals() and file_handler:
            try:
                logging.getLogger().removeHandler(file_handler)
                file_handler.close()
            except Exception as log_e:
                print(f"Error closing file logger: {log_e}")
        total_time = time.time() - start_time_total
        logger.info(f"Processing finished. Status: {results.get('status')}. Total time: {total_time:.2f} seconds.")
        if results.get('error'):
            logger.error(f"Error details: {results.get('error')}")
    return results

def make_path_relative(absolute_path, base_dir):
    """Convert absolute paths to relative paths for YAML output."""
    if not absolute_path:
        return None
    try:
        return os.path.relpath(absolute_path, base_dir)
    except ValueError:
        logging.warning(f"Could not make path relative: {absolute_path}")
        return absolute_path

def main():
    parser = argparse.ArgumentParser(description="WhisperBite: Transcribe & Diarize Audio")
    parser.add_argument("input_path", help="Path to the audio/video file, directory, or URL.")
    parser.add_argument("output_dir", help="Directory to save the results.")
    parser.add_argument("--device", default=None, help="Processing device (cuda or cpu). Overrides settings.yaml if provided.")
    parser.add_argument("--hf_token", help="Hugging Face API token (required for diarization). Overrides settings.yaml if provided.")
    parser.add_argument("--workflow_yaml", required=True, help="Path to a workflow YAML file defining the modular pipeline steps.")
    args = parser.parse_args()

    # --- Load Global Settings ---
    settings_path = "settings.yaml"
    if not os.path.exists(settings_path):
        logger.error(f"Global settings file '{settings_path}' not found. Please create it (see settings.yaml.example).")
        sys.exit(1)
    with open(settings_path, "r") as f:
        config = yaml.safe_load(f)
    # Merge CLI overrides
    if args.device:
        config["device"] = args.device
    if args.hf_token:
        config["hf_token"] = args.hf_token
    config["workflow_yaml"] = args.workflow_yaml

    # --- Handle Input Path (same as before) ---
    input_path = args.input_path
    is_url = input_path.startswith('http://') or input_path.startswith('https://')
    is_dir = os.path.isdir(input_path)
    actual_input_file = None
    temp_dir = None # For downloaded files

    if is_url:
        try:
            logger.info(f"Input is a URL, attempting download...")
            temp_dir = tempfile.mkdtemp(prefix="whisperbite_download_")
            actual_input_file = download_audio(input_path, temp_dir)
            if actual_input_file is None:
                raise Exception("Download failed.")
            logger.info(f"Downloaded audio to temporary file: {actual_input_file}")
            input_basename = sanitize_filename(os.path.splitext(os.path.basename(actual_input_file))[0])
        except Exception as e:
            logger.error(f"Failed to download or process URL {input_path}: {e}")
            if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            sys.exit(1)
    elif is_dir:
        logger.info(f"Input is a directory, searching for the newest compatible file...")
        compatible_extensions = tuple(list(whisper.audio.SUPPORTED_EXTENSIONS) + list(VIDEO_EXTENSIONS))
        files = []
        for f in os.listdir(input_path):
            f_path = os.path.join(input_path, f)
            if os.path.isfile(f_path) and f.lower().endswith(compatible_extensions):
                files.append((f_path, os.path.getmtime(f_path)))
        if not files:
            logger.error(f"No compatible audio/video files found in directory: {input_path}")
            sys.exit(1)
        files.sort(key=lambda x: x[1], reverse=True)
        actual_input_file = files[0][0]
        logger.info(f"Processing newest file in directory: {actual_input_file}")
        input_basename = sanitize_filename(os.path.splitext(os.path.basename(actual_input_file))[0])
    elif os.path.isfile(input_path):
        actual_input_file = input_path
        input_basename = sanitize_filename(os.path.splitext(os.path.basename(actual_input_file))[0])
    else:
        logger.error(f"Input path '{input_path}' is not a valid file, directory, or URL.")
        sys.exit(1)

    # --- Check for Video and Extract Audio (same as before) ---
    if actual_input_file.lower().endswith(VIDEO_EXTENSIONS):
        logger.info("Input is a video file, extracting audio...")
        try:
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp(prefix="whisperbite_extract_")
            extracted_audio_path = os.path.join(temp_dir, f"{input_basename}_extracted_audio.wav")
            cmd = ["ffmpeg", "-y", "-i", actual_input_file, "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", extracted_audio_path]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Extracted audio to: {extracted_audio_path}")
            actual_input_file = extracted_audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio from video: {e}")
            logger.error(f"STDERR: {e.stderr.decode('utf-8')}")
            if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred during audio extraction: {e}")
            if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            sys.exit(1)

    # --- Prepare Output Directory ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"{input_basename}_{timestamp}")
    try:
        os.makedirs(run_output_dir, exist_ok=True)
        logger.info(f"Results will be saved in: {run_output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {run_output_dir}: {e}")
        if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        sys.exit(1)

    # --- Warn if running on CPU for long files ---
    if config.get('device', 'cuda') == 'cpu':
        logger.warning("You are running on CPU. Processing long files may be extremely slow. Use --device cuda if you have a GPU.")

    # --- Run Processing ---
    try:
        results = process_audio(
            input_file=actual_input_file,
            output_dir=run_output_dir,
            preset_name="CLI",
            preset_config=config,
            device=config.get('device', 'cuda'),
            progress_callback=lambda step, status, msg: logger.info(f"Progress: [{status.upper()}] {step} - {msg if msg else ''}")
        )
        logger.info(f"Processing completed. Status: {results.get('status')}")
        if results.get('error'):
            logger.error(f"Error during processing: {results.get('error')}")
    except Exception as e:
        logger.exception("An unexpected error occurred in the main processing workflow.")
        results = {"status": "error", "error": traceback.format_exc()}
    # --- Cleanup ---
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
    sys.exit(0 if results.get("status") == "completed" else 1)

@register_step('per_segment_legacy_pipeline')
def step_per_segment_legacy_pipeline(context, params):
    import yaml
    import shutil
    import soundfile as sf
    import numpy as np
    from collections import defaultdict
    segment_yaml_paths = []
    output_dir = context['output_dir']
    model_name = params.get('whisper_model_size', 'base')
    demucs_model = params.get('demucs_model', 'htdemucs')
    num_speakers = params.get('num_speakers', 2)
    auto_speakers = params.get('auto_speakers', False)
    vad_threshold = params.get('vad_threshold', 0.5)
    enable_word_extraction = params.get('enable_word_extraction', False)
    enable_second_pass = params.get('enable_second_pass', False)
    attempt_sound_detection = params.get('attempt_sound_detection', False)
    segments = context.get('conversation_segments', [])
    per_segment_results = []
    # Group segments by call_id
    calls = defaultdict(list)
    for seg in segments:
        call_id = seg.get('call_id', 0)
        calls[call_id].append(seg)
    for call_id, segs in calls.items():
        # Concatenate all segments for this call into a single audio file
        call_audio_paths = [seg['path'] for seg in segs]
        call_audio_datas = []
        sample_rate = None
        for path in call_audio_paths:
            data, sr = sf.read(path)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError(f"Sample rate mismatch in call {call_id} segments.")
            call_audio_datas.append(data)
        call_audio = np.concatenate(call_audio_datas, axis=0)
        call_dir = os.path.join(output_dir, f'call_{call_id:02d}')
        os.makedirs(call_dir, exist_ok=True)
        call_audio_path = os.path.join(call_dir, f'call_{call_id:02d}_full.wav')
        sf.write(call_audio_path, call_audio, sample_rate)
        # Run Demucs once on the full call audio
        vocals_file, no_vocals_file = separate_vocals_with_demucs(call_audio_path, call_dir, model=demucs_model)
        # For each segment in the call, process using the Demucs outputs for that call
        offset = 0.0
        for seg in segs:
            seg_path = seg['path']
            seg_id = os.path.splitext(os.path.basename(seg_path))[0]
            seg_out_dir = os.path.join(call_dir, seg_id)
            os.makedirs(seg_out_dir, exist_ok=True)
            # Calculate segment start/end relative to the concatenated call audio
            seg_start = offset
            seg_end = offset + (sf.info(seg_path).duration)
            offset = seg_end
            # Extract segment from Demucs vocals for downstream processing
            vocals_seg_path = os.path.join(seg_out_dir, f'{seg_id}_vocals.wav')
            no_vocals_seg_path = os.path.join(seg_out_dir, f'{seg_id}_no_vocals.wav')
            extract_audio_segment(vocals_file, seg_start, seg_end, vocals_seg_path)
            extract_audio_segment(no_vocals_file, seg_start, seg_end, no_vocals_seg_path)
            # Run the old pipeline logic on this segment (using vocals_seg_path)
            result = process_segment_with_old_pipeline(
                vocals_seg_path,
                seg_out_dir,
                model_name,
                False,  # Demucs already done
                num_speakers,
                auto_speakers,
                enable_word_extraction,
                enable_second_pass,
                attempt_sound_detection
            )
            # Write per-segment YAML
            seg_yaml_path = os.path.join(seg_out_dir, f"{seg_id}.yaml")
            with open(seg_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(result, f, allow_unicode=True, sort_keys=False)
            segment_yaml_paths.append(seg_yaml_path)
            per_segment_results.append(result)
    context['segment_yaml_paths'] = segment_yaml_paths
    context['per_segment_results'] = per_segment_results
    return context

@register_step('aggregate_outputs')
def step_aggregate_outputs(context, params):
    import yaml
    output_dir = context['output_dir']
    soundbite_metadata = context.get('soundbite_metadata', [])
    master_index_path = os.path.join(output_dir, 'master_index.yaml')
    master_transcript_yaml_path = os.path.join(output_dir, 'master_transcript.yaml')
    master_transcript_txt_path = os.path.join(output_dir, 'master_transcript.txt')
    if not soundbite_metadata:
        logger.error("[AGGREGATE] No soundbite metadata found. Writing minimal error outputs.")
        error_info = {'error': 'No soundbites or transcripts were produced. See logs for details.'}
        with open(master_index_path, 'w', encoding='utf-8') as f:
            yaml.dump(error_info, f, allow_unicode=True, sort_keys=False)
        with open(master_transcript_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(error_info, f, allow_unicode=True, sort_keys=False)
        with open(master_transcript_txt_path, 'w', encoding='utf-8') as f:
            f.write('ERROR: No soundbites or transcripts were produced. See processing.log for details.\n')
        context['aggregate_outputs_error'] = error_info['error']
        context['master_index_path'] = master_index_path
        context['master_transcript_yaml_path'] = master_transcript_yaml_path
        context['master_transcript_txt_path'] = master_transcript_txt_path
        return context
    # Write master_index.yaml
    with open(master_index_path, 'w', encoding='utf-8') as f:
        yaml.dump({'soundbites': soundbite_metadata}, f, allow_unicode=True, sort_keys=False)
    # Write master_transcript.yaml
    with open(master_transcript_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(soundbite_metadata, f, allow_unicode=True, sort_keys=False)
    # Write master_transcript.txt (with event markers)
    with open(master_transcript_txt_path, 'w', encoding='utf-8') as f:
        for meta in soundbite_metadata:
            speaker = meta.get('speaker', 'UNK')
            start = meta.get('start', 0)
            end = meta.get('end', 0)
            transcript_with_events = meta.get('transcript_with_events', meta.get('transcript', '').strip())
            seq = meta.get('sequence', 0)
            f.write(f"{seq:04d} - {speaker}: [{start:.2f}s - {end:.2f}s] {transcript_with_events}\n\n")
    logger.info(f"[AGGREGATE] Wrote master_index.yaml, master_transcript.yaml, and master_transcript.txt in canonical format with event context.")
    context['master_index_path'] = master_index_path
    context['master_transcript_yaml_path'] = master_transcript_yaml_path
    context['master_transcript_txt_path'] = master_transcript_txt_path
    return context

@register_step('merge_transcripts_advanced')
def step_merge_transcripts_advanced(context, params):
    import numpy as np
    from collections import defaultdict
    import logging
    # Get diarized segments and whole-file transcript
    diarized = context.get('speaker_soundbites', [])
    wholefile = context.get('transcription', {})
    clap_events = context.get('clap_segments', [])
    if not diarized or not wholefile or 'segments' not in wholefile:
        logger.warning("[MERGE_ADV] Missing diarized or whole-file transcript. Skipping advanced merge.")
        context['merged_transcript'] = []
        return context
    diarized_texts = [s['transcription'] for s in diarized if s.get('transcription')]
    diarized_meta = [s for s in diarized if s.get('transcription')]
    whole_segments = wholefile['segments']
    whole_texts = [seg['text'] for seg in whole_segments if seg.get('text')]
    # Load embedding model
    model_name = params.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer(model_name)
    # Compute embeddings
    diarized_emb = model.encode(diarized_texts, convert_to_tensor=True)
    whole_emb = model.encode(whole_texts, convert_to_tensor=True)
    # Compute similarity matrix
    sim_matrix = util.cos_sim(diarized_emb, whole_emb).cpu().numpy()
    threshold = float(params.get('similarity_threshold', 0.8))
    used_whole = set()
    merged = []
    for i, d_meta in enumerate(diarized_meta):
        # Find best match in whole-file transcript
        best_j = np.argmax(sim_matrix[i])
        best_sim = sim_matrix[i][best_j]
        block_start = d_meta.get('start_time', 0)
        block_end = d_meta.get('end_time', 0)
        # Find overlapping CLAP events
        overlapping_events = []
        for event in clap_events:
            event_start = event.get('start', event.get('start_time', 0))
            event_end = event.get('end', event.get('end_time', 0))
            if max(block_start, event_start) < min(block_end, event_end):
                overlapping_events.append(event)
        block = {
            'speaker': d_meta.get('speaker', 'UNK'),
            'start': block_start,
            'end': block_end,
            'text': d_meta.get('transcription', ''),
            'source': 'diarized',
            'word_timestamps': d_meta.get('word_timestamps', []),
            'clap_events': overlapping_events,
        }
        merged.append(block)
        if best_sim >= threshold:
            used_whole.add(best_j)
    # Add whole-file segments not matched to any diarized segment
    for j, seg in enumerate(whole_segments):
        if j not in used_whole:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            # Find overlapping CLAP events
            overlapping_events = []
            for event in clap_events:
                event_start = event.get('start', event.get('start_time', 0))
                event_end = event.get('end', event.get('end_time', 0))
                if max(seg_start, event_start) < min(seg_end, event_end):
                    overlapping_events.append(event)
            merged.append({
                'speaker': 'CONTEXT',
                'start': seg_start,
                'end': seg_end,
                'text': seg.get('text', ''),
                'source': 'wholefile',
                'word_timestamps': seg.get('words', []),
                'clap_events': overlapping_events,
            })
    merged.sort(key=lambda x: x['start'])
    num_with_events = sum(1 for b in merged if b.get('clap_events'))
    logger.info(f"[MERGE_ADV] Merged {len(diarized_meta)} diarized and {len(whole_segments)} whole-file segments into {len(merged)} deduplicated transcript blocks. {num_with_events} blocks have injected CLAP events.")
    context['merged_transcript'] = merged
    return context

def merge_and_deduplicate_segments(segments, max_gap=0.5, min_duration=1.0):
    """
    Merge overlapping or adjacent segments for the same speaker.
    Only keep unique, non-overlapping segments.
    """
    if not segments:
        return []
    # Sort by speaker, then start time
    segments = sorted(segments, key=lambda x: (x['speaker'], x['start']))
    merged = []
    last = None
    for seg in segments:
        if last is None:
            last = seg.copy()
            continue
        if seg['speaker'] == last['speaker'] and seg['start'] <= last['end'] + max_gap:
            # Merge segments
            last['end'] = max(last['end'], seg['end'])
            last['duration'] = last['end'] - last['start']
        else:
            # Only keep if long enough
            if last['end'] - last['start'] >= min_duration:
                merged.append(last)
            last = seg.copy()
    # Add the last segment
    if last and last['end'] - last['start'] >= min_duration:
        merged.append(last)
    return merged

@register_step('build_conversation_blocks')
def step_build_conversation_blocks(context, params):
    import os
    import soundfile as sf
    diarization = context.get('diarization', [])
    vocals_path = context.get('demucs_vocals_path', context.get('normalized_audio_path'))
    output_dir = context.get('output_dir', '.')
    clap_events = context.get('clap_events', [])
    if not diarization or not vocals_path or not os.path.exists(vocals_path):
        logger.error("[BUILD_BLOCKS] Missing diarization or vocals audio. Cannot build conversation blocks.")
        context['conversation_blocks'] = []
        return context
    # --- Deduplicate/merge segments before block creation ---
    logger.info(f"[BUILD_BLOCKS] Segments before deduplication: {len(diarization)}")
    merged_segments = merge_and_deduplicate_segments(diarization, max_gap=0.5, min_duration=1.0)
    logger.info(f"[BUILD_BLOCKS] Segments after deduplication/merging: {len(merged_segments)}")
    # Load audio to get sample rate
    try:
        with sf.SoundFile(vocals_path, 'r') as infile:
            sample_rate = infile.samplerate
    except Exception as e:
        logger.error(f"[BUILD_BLOCKS] Failed to read audio file {vocals_path}: {e}")
        context['conversation_blocks'] = []
        return context
    blocks = []
    for i, seg in enumerate(merged_segments):
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        speaker = seg.get('speaker', 'UNK')
        block_id = f"block_{i:04d}"
        # Extract segment audio
        segment_audio_path = os.path.join(output_dir, 'segments', f"{block_id}.wav")
        os.makedirs(os.path.dirname(segment_audio_path), exist_ok=True)
        try:
            extract_audio_segment(vocals_path, start, end, segment_audio_path)
        except Exception as e:
            logger.error(f"[BUILD_BLOCKS] Failed to extract segment {block_id}: {e}")
            continue
        # Associate CLAP events with this block
        block_clap_events = [e for e in clap_events if e['start'] < end and e['end'] > start]
        blocks.append({
            'block_id': block_id,
            'start_time': start,
            'end_time': end,
            'speaker': speaker,
            'segment_audio_path': segment_audio_path,
            'clap_events': block_clap_events
        })
    logger.info(f"[BUILD_BLOCKS] Created {len(blocks)} conversation blocks after deduplication/merging.")
    context['conversation_blocks'] = blocks
    context['results']['conversation_blocks'] = blocks
    return context

# Add robust error logging to downstream steps (whisper, diarize_and_split_speakers, soundbite, write_transcripts)
# (Add at the top of each step)

def robust_check_and_log(context, key, step_name):
    if not context.get(key):
        logger.error(f"[{step_name}] Missing required context key: '{key}'. Step will do nothing.")
        return False
    return True

# Patch whisper step
old_step_whisper = step_whisper
@register_step('whisper')
def step_whisper(context, params):
    if not robust_check_and_log(context, 'conversation_blocks', 'WHISPER'):
        return context
    return old_step_whisper(context, params)

# Patch diarize_and_split_speakers step
old_step_diarize_and_split_speakers = step_diarize_and_split_speakers
@register_step('diarize_and_split_speakers')
def step_diarize_and_split_speakers(context, params):
    if not robust_check_and_log(context, 'conversation_blocks', 'DIARIZE_SPLIT'):
        return context
    return old_step_diarize_and_split_speakers(context, params)

# Patch soundbite step
old_step_soundbite = step_soundbite
@register_step('soundbite')
def step_soundbite(context, params):
    if not robust_check_and_log(context, 'speaker_soundbites', 'SOUNDBITE'):
        return context
    return old_step_soundbite(context, params)

# Patch write_transcripts step
old_step_write_transcripts = step_write_transcripts
@register_step('write_transcripts')
def step_write_transcripts(context, params):
    if not robust_check_and_log(context, 'speaker_soundbites', 'WRITE_TRANSCRIPTS'):
        return context
    return old_step_write_transcripts(context, params)

@register_step('vad_and_clap_chunking')
def step_vad_and_clap_chunking(context, params):
    import soundfile as sf
    from event_detection import run_clap_event_detection
    import numpy as np
    import yaml
    audio_path = context.get('demucs_vocals_path', context.get('normalized_audio_path'))
    output_dir = context.get('output_dir', '.')
    if not audio_path or not os.path.exists(audio_path):
        logger.error('[VAD_CLAP_CHUNK] No audio file found for chunking.')
        context['vad_chunks'] = []
        return context
    # --- VAD chunking ---
    vad_threshold = params.get('vad_threshold', 0.5)
    max_vad_chunk = params.get('max_vad_chunk', 60.0)
    vad_regions = detect_speech_regions_vad(audio_path, threshold=vad_threshold)
    # Enforce max chunk duration
    vad_chunks = []
    for region in vad_regions:
        start, end = region['start'], region['end']
        while end - start > max_vad_chunk:
            vad_chunks.append({'start': start, 'end': start + max_vad_chunk})
            start += max_vad_chunk
        vad_chunks.append({'start': start, 'end': end})
    logger.info(f"[VAD_CLAP_CHUNK] {len(vad_chunks)} VAD chunks created (max {max_vad_chunk}s each).")
    # --- CLAP event detection on sub-chunks ---
    clap_model_id = params.get('clap_model_id', 'laion/clap-htsat-fused')
    clap_threshold = params.get('clap_threshold', 0.5)
    clap_chunk = params.get('clap_chunk', 5.0)
    clap_overlap = params.get('clap_overlap', 2.0)
    clap_prompts = params.get('clap_prompts', ["phone ringing", "music", "applause", "speech"])
    device = context.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, str):
        device = torch.device(device)
    logger.info(f"[VAD_CLAP_CHUNK] Loading CLAP model {clap_model_id} on {device}...")
    clap_model, clap_processor = load_clap_model_and_processor(clap_model_id, device)
    if not clap_model or not clap_processor:
        logger.error("[VAD_CLAP_CHUNK] Failed to load CLAP model/processor. Skipping CLAP event detection.")
        for chunk in vad_chunks:
            chunk['clap_events'] = []
        context['vad_chunks'] = vad_chunks
        return context
    # Load audio
    audio_data, sr = sf.read(audio_path, dtype='float32')
    if sr != 48000:
        logger.warning(f"[VAD_CLAP_CHUNK] Resampling audio from {sr}Hz to 48000Hz for CLAP.")
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=48000)
        sr = 48000
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    all_clap_events = []
    for chunk in vad_chunks:
        chunk_start, chunk_end = chunk['start'], chunk['end']
        chunk_events = []
        # Subdivide chunk for CLAP
        sub_start = chunk_start
        while sub_start < chunk_end:
            sub_end = min(sub_start + clap_chunk, chunk_end)
            # Extract sub-chunk audio
            start_frame = int(sub_start * sr)
            end_frame = int(sub_end * sr)
            sub_audio = audio_data[start_frame:end_frame]
            # Run CLAP event detection
            try:
                detected_events = run_clap_event_detection(
                    audio_data=sub_audio,
                    sample_rate=sr,
                    clap_model=clap_model,
                    clap_processor=clap_processor,
                    device=device,
                    target_events=clap_prompts,
                    threshold=clap_threshold,
                    chunk_duration=clap_chunk,
                    min_gap=1.0
                )
                # Offset event times by sub_start
                for event_type, events in detected_events.items():
                    for e in events:
                        event = e.copy()
                        event['type'] = event_type
                        event['start'] += sub_start
                        event['end'] += sub_start
                        chunk_events.append(event)
                        all_clap_events.append(event)
            except Exception as e:
                logger.error(f"[VAD_CLAP_CHUNK] CLAP failed on sub-chunk {sub_start:.2f}-{sub_end:.2f}: {e}")
            sub_start += clap_chunk - clap_overlap
        # Merge/aggregate events for the chunk
        chunk['clap_events'] = chunk_events
        logger.info(f"[VAD_CLAP_CHUNK] Chunk {chunk_start:.2f}-{chunk_end:.2f}s: {len(chunk_events)} CLAP events detected.")
    context['vad_chunks'] = vad_chunks
    context['clap_events'] = all_clap_events
    # Write CLAP events to disk
    try:
        clap_events_path = os.path.join(output_dir, 'clap_events.yaml')
        with open(clap_events_path, 'w') as f:
            yaml.safe_dump({'clap_events': all_clap_events}, f)
        logger.info(f"[VAD_CLAP_CHUNK] Wrote {len(all_clap_events)} CLAP events to {clap_events_path}")
    except Exception as e:
        logger.error(f"[VAD_CLAP_CHUNK] Failed to write CLAP events to disk: {e}")
    return context

if __name__ == "__main__":
    main()
