import os
import logging
import torch
import librosa
import numpy as np
from transformers import ClapModel, ClapProcessor
from pydub import AudioSegment
import soundfile as sf
from typing import List, Dict, Tuple, Optional
import subprocess
import time
from tqdm import tqdm
import psutil
import math
import threading
import re

# <<< Moved logger definition here >>>
logger = logging.getLogger(__name__)

def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in the system."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        logging.error("ffmpeg not found. Please install ffmpeg to process audio files.")
        return False

def validate_audio_file(audio_path: str) -> bool:
    """Validate that the audio file exists and is not empty."""
    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found: {audio_path}")
        return False
    if os.path.getsize(audio_path) == 0:
        logging.error(f"Audio file is empty: {audio_path}")
        return False
    return True

def cleanup_gpu_memory():
    """Clean up GPU memory if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Constants
# Model chosen based on general availability and performance. Others could be tested.
CLAP_MODEL_ID = "laion/clap-htsat-unfused" 
# Typical sample rate for CLAP models
CLAP_SAMPLE_RATE = 48000 
# TODO: Make these configurable via args/UI
DEFAULT_CHUNK_DURATION_S = 5.0 # Process audio in 5-second chunks
DEFAULT_DETECTION_THRESHOLD = 0.7 # Confidence threshold for detection

# Call analysis specific defaults
DEFAULT_CALL_CHUNK_DURATION = 5.0  # seconds
DEFAULT_CALL_THRESHOLD = 0.7

# Define the text prompts for sounds we want to detect
# Keep these somewhat generic for better model generalization, unless specific sounds are crucial.
TARGET_SOUND_PROMPTS = [
    "speech", 
    "music", 
    "telephone ringing", 
    "applause", 
    "dog barking",
    "doorbell",
    "siren",
    "alarm clock" 
    # Add more general or specific prompts as needed
]

# Call-specific sound prompts
CALL_ANALYSIS_PROMPTS = [
    "telephone ringing",
    "phone ringing",
    "dial tone",
    "busy signal",
    "phone beep",
    "phone notification",
    "phone alert"
]

# Vocal cues to detect in the vocals track
VOCAL_CUES_FOR_ANALYSIS = [
    "laughter",
    "coughing",
    "throat clearing",
    "sighing",
    "breathing",
    "background chatter"
]

# Global variables to hold the loaded model and processor
clap_model = None
clap_processor = None
device = None

def load_clap_model():
    """Loads the CLAP model and processor from Hugging Face Transformers."""
    global clap_model, clap_processor, device
    if clap_model is None or clap_processor is None:
        try:
            logging.info(f"Loading CLAP model ({CLAP_MODEL_ID}) and processor...")
            # Determine device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logging.info("CUDA available, using GPU.")
            else:
                device = torch.device("cpu")
                logging.info("CUDA not available, using CPU.")
            
            clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
            clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
            clap_model.eval() # Set model to evaluation mode
            logging.info("CLAP model and processor loaded successfully.")
            
        except Exception as e:
            logging.error(f"Failed to load CLAP model or processor: {e}", exc_info=True)
            clap_model = None
            clap_processor = None
            device = None
            return False
    return clap_model is not None and clap_processor is not None

def cut_audio_at_detections(audio_path: str, detected_events: list, output_dir: str, min_segment_duration: float = 1.0) -> list:
    """
    Cut audio file at points where high-confidence sounds are detected.
    
    Args:
        audio_path (str): Path to the audio file to cut
        detected_events (list): List of detected sound events with start/end times
        output_dir (str): Directory to save cut segments
        min_segment_duration (float): Minimum duration for a segment to be saved
        
    Returns:
        list: List of dictionaries containing segment information
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        total_duration = len(audio) / 1000.0  # Duration in seconds
        
        # Sort events by start time
        detected_events.sort(key=lambda x: x['start'])
        
        segments = []
        current_start = 0
        
        # Create segments directory
        segments_dir = os.path.join(output_dir, "cut_segments")
        os.makedirs(segments_dir, exist_ok=True)
        
        # Process each detection point
        for i, event in enumerate(detected_events):
            event_start = event['start']
            event_end = event['end']
            
            # Only cut if we have a reasonable segment duration
            if event_start - current_start >= min_segment_duration:
                # Extract segment
                segment_audio = audio[current_start*1000:event_start*1000]
                
                # Generate segment filename
                segment_name = f"segment_{i:04d}_{current_start:.2f}_{event_start:.2f}.wav"
                segment_path = os.path.join(segments_dir, segment_name)
                
                # Export segment
                segment_audio.export(segment_path, format="wav")
                
                # Store segment information
                segments.append({
                    'start': current_start,
                    'end': event_start,
                    'duration': event_start - current_start,
                    'path': segment_path,
                    'trigger_event': event['text'],
                    'trigger_confidence': event['confidence']
                })
            
            # Update start point for next segment
            current_start = event_end
        
        # Handle final segment if needed
        if total_duration - current_start >= min_segment_duration:
            segment_audio = audio[current_start*1000:]
            segment_name = f"segment_{len(segments):04d}_{current_start:.2f}_{total_duration:.2f}.wav"
            segment_path = os.path.join(segments_dir, segment_name)
            segment_audio.export(segment_path, format="wav")
            
            segments.append({
                'start': current_start,
                'end': total_duration,
                'duration': total_duration - current_start,
                'path': segment_path,
                'trigger_event': "end_of_file",
                'trigger_confidence': 1.0
            })
        
        return segments
        
    except Exception as e:
        logging.error(f"Error cutting audio at detections: {e}", exc_info=True)
        return []

def log_gpu_stats():
    """Log GPU memory usage statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logging.info(f"GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

def debug_tensor(name: str, tensor: torch.Tensor):
    """Debug helper to log tensor details."""
    logging.info(f"{name}:")
    logging.info(f"  Shape: {tensor.shape}")
    logging.info(f"  Device: {tensor.device}")
    logging.info(f"  Dtype: {tensor.dtype}")
    logging.info(f"  Memory: {tensor.element_size() * tensor.nelement() / 1024**2:.2f}MB")
    if torch.isnan(tensor).any():
        logging.warning(f"  Contains NaN values!")
    if torch.isinf(tensor).any():
        logging.warning(f"  Contains Inf values!")

def debug_dict(name: str, d: dict):
    """Debug helper to log dictionary contents."""
    logging.info(f"{name} keys: {list(d.keys())}")
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            debug_tensor(f"{name}[{k}]", v)
        elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            logging.info(f"{name}[{k}] is a sequence of {len(v)} tensors")
            debug_tensor(f"{name}[{k}][0]", v[0])

def process_audio_chunk(model, processor, audio_data, sampling_rate, device, debug=True):
    """Process a single audio chunk with detailed debugging."""
    try:
        if debug:
            logging.info("Audio input stats:")
            logging.info(f"  Shape: {audio_data.shape}")
            logging.info(f"  Sample rate: {sampling_rate}")
            logging.info(f"  Duration: {len(audio_data)/sampling_rate:.2f}s")
            logging.info(f"  Min: {audio_data.min():.3f}, Max: {audio_data.max():.3f}")
            if np.isnan(audio_data).any():
                logging.error("Audio data contains NaN values!")
            if np.isinf(audio_data).any():
                logging.error("Audio data contains Inf values!")

        logging.info("Creating inputs with processor...")
        inputs = processor(
            audios=audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )
        if debug:
            debug_dict("Processor inputs", inputs)
        
        logging.info("Moving inputs to device...")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if debug:
            debug_dict("Device inputs", inputs)

        logging.info("Running model inference...")
        with torch.no_grad():
            try:
                audio_features = model.get_audio_features(**inputs)
                if debug:
                    debug_tensor("Audio features", audio_features)
                return audio_features
            except Exception as e:
                logging.error(f"Model inference failed: {str(e)}")
                logging.error("Model state:", exc_info=True)
                logging.error(f"Model device: {next(model.parameters()).device}")
                logging.error(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
                raise

    except Exception as e:
        logging.error(f"Error in process_audio_chunk: {str(e)}")
        logging.error("Full traceback:", exc_info=True)
        raise

def detect_and_cut_audio(
    audio_path: str,
    output_dir: str,
    chunk_duration_s: float = DEFAULT_CALL_CHUNK_DURATION,
    overlap_s: float = 1.0,
    threshold: float = DEFAULT_CALL_THRESHOLD,
    target_prompts: List[str] = CALL_ANALYSIS_PROMPTS,
    enable_cutting: bool = True,
    max_chunk_process_time: int = 30,
    stop_event: Optional[threading.Event] = None
) -> Tuple[List[Dict], Optional[List[Dict]]]:
    """Enhanced version with detailed CLAP debugging."""
    try:
        # Validate prerequisites
        if not check_ffmpeg_available():
            return [], None
        
        if not validate_audio_file(audio_path):
            return [], None
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        cut_segments_dir = os.path.join(output_dir, "cut_segments")
        if enable_cutting:
            os.makedirs(cut_segments_dir, exist_ok=True)
        
        # Load the audio file and log initial stats
        logging.info(f"Loading audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        total_duration_s = duration_ms / 1000.0
        
        logging.info(f"Audio loaded successfully. Duration: {total_duration_s:.1f}s")
        logging.info(f"Chunk size: {chunk_duration_s}s with {overlap_s}s overlap")
        log_gpu_stats()
        
        # Initialize CLAP model and processor
        logging.info("Initializing CLAP model and processor...")
        t_start = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        try:
            model = ClapModel.from_pretrained(CLAP_MODEL_ID)
            logging.info(f"Model loaded in {time.time() - t_start:.1f}s")
            
            processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
            logging.info(f"Processor loaded in {time.time() - t_start:.1f}s")
            
            model = model.to(device)
            logging.info(f"Model moved to {device} in {time.time() - t_start:.1f}s")
            model.eval()
        except Exception as e:
            logging.error(f"Failed to initialize CLAP model: {str(e)}")
            logging.error("Full traceback:", exc_info=True)
            return [], None
        
        # Pre-compute text features once
        logging.info("Pre-computing text features...")
        try:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                text_inputs = processor(text=target_prompts, return_tensors="pt", padding=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    text_features = model.get_text_features(**text_inputs)
            logging.info("Text features pre-computed successfully")
            debug_tensor("Text features", text_features)
            log_gpu_stats()
        except Exception as e:
            logging.error(f"Failed to pre-compute text features: {str(e)}")
            logging.error("Full traceback:", exc_info=True)
            return [], None
        
        # Calculate chunks with overlap
        chunk_size_ms = int(chunk_duration_s * 1000)
        overlap_ms = int(overlap_s * 1000)
        step_size_ms = chunk_size_ms - overlap_ms
        total_chunks = math.ceil((duration_ms - overlap_ms) / step_size_ms)
        
        detected_events = []
        cut_segments = [] if enable_cutting else None
        current_ms = 0
        
        # Process chunks with progress bar
        with tqdm(total=total_chunks, desc="Processing audio chunks") as pbar:
            while current_ms < duration_ms:
                if stop_event and stop_event.is_set():
                    logging.info("Stop requested, cleaning up...")
                    break
                
                chunk_start_time = time.time()
                
                # Define chunk boundaries
                end_ms = min(current_ms + chunk_size_ms, duration_ms)
                logging.info(f"\nProcessing chunk {current_ms/1000:.1f}s to {end_ms/1000:.1f}s")
                
                chunk = audio[current_ms:end_ms]
                
                # Export chunk for processing
                chunk_path = os.path.join(output_dir, f"temp_chunk_{current_ms}.wav")
                logging.info(f"Exporting chunk to {chunk_path}")
                try:
                    chunk.export(
                        chunk_path,
                        format="wav",
                        parameters=[
                            "-ar", str(CLAP_SAMPLE_RATE),
                            "-acodec", "pcm_s16le"
                        ]
                    )
                except Exception as e:
                    logging.error(f"Failed to export chunk: {str(e)}")
                    current_ms += step_size_ms
                    pbar.update(1)
                    continue
                
                try:
                    # Load and process audio
                    logging.info("Loading chunk with soundfile...")
                    audio_data, sr = sf.read(chunk_path)
                    if sr != CLAP_SAMPLE_RATE:
                        logging.info(f"Resampling from {sr}Hz to {CLAP_SAMPLE_RATE}Hz")
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=CLAP_SAMPLE_RATE)
                    
                    # Process with GPU and automatic mixed precision
                    logging.info("Processing chunk with CLAP model...")
                    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        # Process audio features
                        audio_features = process_audio_chunk(
                            model=model,
                            processor=processor,
                            audio_data=audio_data,
                            sampling_rate=CLAP_SAMPLE_RATE,
                            device=device,
                            debug=True
                        )
                        
                        if audio_features is None:
                            logging.error("Failed to get audio features, skipping chunk")
                            current_ms += step_size_ms
                            pbar.update(1)
                            continue
                        
                        # Calculate similarity scores
                        logging.info("Calculating similarity scores...")
                        similarity = torch.nn.functional.cosine_similarity(
                            audio_features[:, None],
                            text_features[None, :],
                            dim=-1
                        )
                        debug_tensor("Similarity scores", similarity)
                        
                        # Find matches above threshold
                        matches = torch.where(similarity > threshold)
                        for audio_idx, text_idx in zip(*matches):
                            event_time = current_ms / 1000.0 + float(audio_idx) * (chunk_duration_s / audio_features.shape[0])
                            confidence = float(similarity[audio_idx, text_idx])
                            detected_events.append({
                                'type': target_prompts[text_idx],
                                'start': event_time,
                                'confidence': confidence,
                                'chunk_start': current_ms / 1000.0,
                                'chunk_end': end_ms / 1000.0
                            })
                            logging.info(f"Detected '{target_prompts[text_idx]}' at {event_time:.1f}s with confidence {confidence:.3f}")
                
                except Exception as e:
                    logging.error(f"Error processing chunk at {current_ms}ms: {str(e)}")
                    logging.error("Full traceback:", exc_info=True)
                
                finally:
                    # Cleanup
                    try:
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
                    except Exception as e:
                        logging.warning(f"Failed to remove temporary file {chunk_path}: {str(e)}")
                    
                    try:
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logging.warning(f"Failed to clear CUDA cache: {str(e)}")
                    
                    logging.info("Chunk cleanup completed")
                
                # Check if chunk processing took too long
                chunk_process_time = time.time() - chunk_start_time
                if chunk_process_time > max_chunk_process_time:
                    logging.warning(f"Chunk processing took {chunk_process_time:.1f}s (limit: {max_chunk_process_time}s)")
                
                # Update progress
                current_ms += step_size_ms
                pbar.update(1)
                pbar.set_postfix({
                    'Events': len(detected_events),
                    'Time/Chunk': f"{chunk_process_time:.1f}s"
                })
                log_gpu_stats()
        
        # Sort events by time
        detected_events.sort(key=lambda x: x['start'])
        
        # Cut audio at detection points if enabled
        if enable_cutting and detected_events:
            logging.info("Cutting audio at detection points...")
            last_end = 0
            
            for i, event in enumerate(detected_events):
                start_time = last_end
                end_time = int(event['start'] * 1000)  # Convert to milliseconds
                
                if end_time > start_time:
                    segment = audio[start_time:end_time]
                    segment_name = f"segment_{i:03d}_{start_time}_{end_time}.wav"
                    segment_path = os.path.join(cut_segments_dir, segment_name)
                    
                    try:
                        # Export segment
                        segment.export(
                            segment_path,
                            format="wav",
                            parameters=[
                                "-ar", str(CLAP_SAMPLE_RATE),
                                "-acodec", "pcm_s16le"
                            ]
                        )
                        
                        cut_segments.append({
                            'index': i,
                            'path': segment_path,
                            'start': start_time / 1000.0,
                            'end': end_time / 1000.0,
                            'duration': (end_time - start_time) / 1000.0
                        })
                    except Exception as e:
                        logging.error(f"Failed to export segment {i}: {str(e)}")
                
                last_end = end_time
            
            # Handle final segment
            if last_end < len(audio):
                segment = audio[last_end:]
                segment_name = f"segment_{len(detected_events):03d}_{last_end}_{len(audio)}.wav"
                segment_path = os.path.join(cut_segments_dir, segment_name)
                
                try:
                    segment.export(
                        segment_path,
                        format="wav",
                        parameters=[
                            "-ar", str(CLAP_SAMPLE_RATE),
                            "-acodec", "pcm_s16le"
                        ]
                    )
                    
                    cut_segments.append({
                        'index': len(detected_events),
                        'path': segment_path,
                        'start': last_end / 1000.0,
                        'end': len(audio) / 1000.0,
                        'duration': (len(audio) - last_end) / 1000.0
                    })
                except Exception as e:
                    logging.error(f"Failed to export final segment: {str(e)}")
        
        return detected_events, cut_segments
    
    except Exception as e:
        logging.error(f"Error in detect_and_cut_audio: {str(e)}")
        logging.error("Full traceback:", exc_info=True)
        return [], None
    
    finally:
        cleanup_gpu_memory()

def detect_sound_events(
    audio_path: str, 
    chunk_duration_s: float = DEFAULT_CHUNK_DURATION_S, 
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
    target_prompts: list[str] = None
    ):
    """
    Detects specified sound events in an audio file using the CLAP model.

    Args:
        audio_path (str): Path to the audio file (e.g., no_vocals.wav).
        chunk_duration_s (float): Duration of audio chunks to process in seconds.
        threshold (float): Confidence threshold (0.0 to 1.0) for classifying a sound event.
        target_prompts (list[str], optional): A list of text descriptions for sounds to detect. 
                                             Defaults to TARGET_SOUND_PROMPTS.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              detected sound event segment with keys 'start', 'end', 'label' 
              (the detected sound prompt), and 'confidence'. Returns an empty 
              list if the model fails to load or no events are detected.
    """
    if not load_clap_model():
        logging.error("CLAP model is not available. Skipping sound detection.")
        return []

    logging.info(f"Starting sound detection for: {audio_path}")
    logging.info(f"Parameters: Chunk Duration={chunk_duration_s}s, Threshold={threshold}")
    
    detected_events = []
    prompts_to_use = target_prompts if target_prompts else TARGET_SOUND_PROMPTS
    logging.info(f"Using target prompts: {prompts_to_use}")

    global device # Access the device determined by load_clap_model

    try:
        # Load audio using librosa, assuming it's already 48kHz
        logging.info(f"Loading audio with librosa (expecting 48kHz): {audio_path}")
        t_load_start = time.time()
        audio_data, sr = librosa.load(audio_path, sr=None, mono=True) # Load native rate
        logging.info(f"Successfully loaded {audio_path} via librosa in {time.time() - t_load_start:.2f}s, sample rate {sr}")
        
        # --- Add Assertion --- 
        assert sr == CLAP_SAMPLE_RATE, \
            f"[SoundDetector] Expected sample rate {CLAP_SAMPLE_RATE} but received {sr}. Resample upstream."
        # --- End Assertion ---
        
        duration = len(audio_data) / sr
        logging.info(f"Audio duration: {duration:.1f}s")

        if audio_data.size == 0:
            logging.error("Loaded audio data is empty.")
            return []

        # Process audio in chunks
        chunk_size = int(chunk_duration_s * CLAP_SAMPLE_RATE)
        num_chunks = int(np.ceil(len(audio_data) / chunk_size))

        logging.info(f"Processing {num_chunks} chunks...")
        
        for i in tqdm(range(num_chunks), desc="Detecting sounds"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
            
            # Check for very short chunks (can happen at the end)
            if len(chunk) < CLAP_SAMPLE_RATE * 0.1: # Skip if less than 100ms
                continue
                
            # Process the chunk
            try:
                # Process with CLAP
                inputs = clap_processor(
                    text=prompts_to_use, 
                    audios=[chunk], # Processor expects a list of audio arrays
                    return_tensors="pt", 
                    padding=True, 
                    sampling_rate=CLAP_SAMPLE_RATE
                ).to(device)

                with torch.no_grad():
                    # <<< Log Devices >>>
                    logging.debug(f"[SOUNDPROC] Model device: {next(clap_model.parameters()).device}")
                    logging.debug(f"[SOUNDPROC] Input device: {inputs['input_ids'].device}") # Key might be input_ids or similar
                    # <<< End Log >>>
                    outputs = clap_model(**inputs)
                
                # logits_per_audio: [batch_size, num_prompts] 
                # In our case batch_size is 1 (one chunk)
                logits_per_audio = outputs.logits_per_audio 
                
                # Convert logits to probabilities (0-1 range)
                # Handle potential scalar output by ensuring we have the right dimensions
                probs = torch.sigmoid(logits_per_audio)
                if len(probs.shape) == 0:  # scalar
                    probs = probs.unsqueeze(0)  # Add a dimension
                elif len(probs.shape) == 2:  # [batch, num_prompts]
                    probs = probs.squeeze(0)  # Remove batch dimension
                probs = probs.cpu().numpy()

                # Check against threshold for each prompt
                for prompt_idx, prompt_text in enumerate(prompts_to_use):
                    try:
                        confidence = float(probs[prompt_idx] if prompt_idx < len(probs) else probs)
                        if confidence >= threshold:
                            logging.debug(f"Detected '{prompt_text}' (Conf: {confidence:.3f}) at {start_idx/CLAP_SAMPLE_RATE:.2f}s - {end_idx/CLAP_SAMPLE_RATE:.2f}s")
                            detected_events.append({
                                'speaker': 'SOUND', # Keep consistent label for sound events
                                'start': start_idx/CLAP_SAMPLE_RATE,
                                'end': end_idx/CLAP_SAMPLE_RATE,
                                'text': f"{prompt_text}", # Just the label, confidence is separate now
                                'confidence': confidence, # Store confidence
                                'audio_file': None, # Keep structure consistent if needed elsewhere
                                'transcript_file': None,
                                'sequence': i # Maintain sequence if helpful
                            })
                    except (IndexError, TypeError) as e:
                        logging.warning(f"Error processing probability for prompt '{prompt_text}': {e}")
                        continue
            
            except Exception as chunk_e:
                logging.error(f"Error processing chunk {i} ({start_idx/CLAP_SAMPLE_RATE:.2f}s-{end_idx/CLAP_SAMPLE_RATE:.2f}s): {chunk_e}", exc_info=True)
                continue # Skip to next chunk on error

    except Exception as e:
        logging.error(f"Failed during sound detection for {audio_path}: {e}", exc_info=True)
        return [] # Return empty list on major failure

    logging.info(f"Finished CLAP sound detection. Found {len(detected_events)} potential events.")
    
    # Optional: Add merging logic for consecutive identical events here if needed
    # For now, returning raw chunk-level detections. Merging can be complex.

    return detected_events

def cut_audio_between_events(audio_path: str, all_events: Dict[str, List[Dict]], output_dir: str,
                           start_types=("ringing phone",), end_types=("hang-up tones",),
                           min_duration_s=2.0, padding_ms=500) -> List[Dict]:
    """
    Cuts audio into segments based on pairs of end/start events.
    Extracts audio *between* the end of an 'end_type' event and the start of the *next* 'start_type' event.

    Args:
        audio_path (str): Path to the original audio file.
        all_events (Dict[str, List[Dict]]): Dictionary of detected events from Pass 1, 
                                           e.g., {'ringing phone': [...], 'hang-up tones': [...]}.
        output_dir (str): Directory to save the cut conversation segments.
        start_types (tuple): Event types marking the beginning of a segment boundary.
        end_types (tuple): Event types marking the end of a segment boundary.
        min_duration_s (float): Minimum duration in seconds for a segment to be saved.
        padding_ms (int): Padding in milliseconds added after the end_event and before the start_event.

    Returns:
        List[Dict]: List of dictionaries for each saved segment, containing 
                    {'path': str, 'start': float, 'end': float, 'duration': float}.
    """
    logger.info(f"Cutting audio between events: start_types={start_types}, end_types={end_types}")
    os.makedirs(output_dir, exist_ok=True)
    
    relevant_events = []
    for event_type, events_list in all_events.items():
        if event_type in start_types or event_type in end_types:
            relevant_events.extend(events_list)

    if not relevant_events:
        logger.warning("No relevant start/end events found for cutting.")
        return []

    # Sort all relevant events by start time
    relevant_events.sort(key=lambda x: x['start'])

    saved_segments = []
    try:
        audio = AudioSegment.from_file(audio_path)
        audio_duration_ms = len(audio)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        segment_index = 0

        last_end_event = None
        for i, current_event in enumerate(relevant_events):
            # Look for an end event
            if current_event['type'] in end_types:
                last_end_event = current_event
                logger.debug(f"Found potential end event: {last_end_event['type']} at {last_end_event['start']:.2f}s")
                continue # Move to the next event to find the corresponding start
            
            # If we have found an end event, look for the *next* start event
            if last_end_event and current_event['type'] in start_types:
                start_event = current_event
                logger.debug(f"Found potential start event: {start_event['type']} at {start_event['start']:.2f}s after end event at {last_end_event['start']:.2f}s")
                
                # Define cut boundaries with padding
                cut_start_ms = int(last_end_event['end'] * 1000) + padding_ms
                cut_end_ms = int(start_event['start'] * 1000) - padding_ms

                # Ensure boundaries are valid and start < end
                cut_start_ms = max(0, cut_start_ms)
                cut_end_ms = min(audio_duration_ms, cut_end_ms)
                
                segment_duration_s = (cut_end_ms - cut_start_ms) / 1000.0

                if segment_duration_s >= min_duration_s:
                    logger.info(f"  -> Extracting segment {segment_index}: {cut_start_ms/1000.0:.2f}s - {cut_end_ms/1000.0:.2f}s (Duration: {segment_duration_s:.2f}s)")
                    segment_audio = audio[cut_start_ms:cut_end_ms]
                    
                    segment_filename = f"{segment_index:04d}_conv_{int(cut_start_ms/1000.0)}_{int(cut_end_ms/1000.0)}.wav"
                    segment_path = os.path.join(output_dir, segment_filename)
                    
                    # Export the segment
                    segment_audio.export(segment_path, format="wav")
                    
                    saved_segments.append({
                        'path': segment_path,
                        'start': cut_start_ms / 1000.0,
                        'end': cut_end_ms / 1000.0,
                        'duration': segment_duration_s
                    })
                    segment_index += 1
                else:
                    logger.debug(f"  -> Skipping segment between {last_end_event['end']:.2f}s and {start_event['start']:.2f}s: Duration {segment_duration_s:.2f}s < {min_duration_s}s")

                # Reset last_end_event, as we've found its corresponding start and cut the segment
                last_end_event = None 
            
            # If the current event is a start event but we haven't seen an end event yet, 
            # it might mark the beginning of the audio before the first hangup, 
            # or it might be an isolated start. Handled by requiring last_end_event to be set.

    except Exception as e:
        logger.error(f"Error during audio cutting between events: {e}", exc_info=True)
        return [] # Return empty list on error

    logger.info(f"Finished cutting between events. Saved {len(saved_segments)} segments.")
    return saved_segments

def extract_soundbites(segment_audio_path: str, segment_annotations: Dict[str, List[Dict]], 
                       output_dir: str, base_filename: str, 
                       min_duration_s=0.2, padding_ms=150, confidence_threshold=0.3) -> List[str]:
    """
    Extracts small audio clips (soundbites) based on Pass 2 CLAP detections.

    Args:
        segment_audio_path (str): Path to the conversation segment audio file.
        segment_annotations (Dict[str, List[Dict]]): Detections from Pass 2 CLAP for this segment.
                                                     e.g., {'speech': [...], 'laughter': [...]}
        output_dir (str): Directory to save extracted soundbites.
        base_filename (str): Base name derived from the segment file (used for soundbite naming).
        min_duration_s (float): Minimum duration for an extracted soundbite.
        padding_ms (int): Padding added around the detected event time for extraction.
        confidence_threshold (float): Minimum confidence for a detection to be extracted.

    Returns:
        List[str]: List of paths to the created soundbite files.
    """
    logger.info(f"Extracting soundbites for {os.path.basename(segment_audio_path)}...")
    os.makedirs(output_dir, exist_ok=True)
    soundbite_paths = []
    soundbite_index = 0

    try:
        segment_audio = AudioSegment.from_file(segment_audio_path)
        segment_duration_ms = len(segment_audio)
    except Exception as e:
        logger.error(f"Could not load segment audio {segment_audio_path} for soundbite extraction: {e}")
        return []

    for event_type, detections in segment_annotations.items():
        for detection in detections:
            try:
                confidence = detection.get('confidence', 0)
                if confidence < confidence_threshold:
                    continue

                start_s = detection['start']
                end_s = detection['end']
                duration_s = end_s - start_s

                if duration_s < min_duration_s:
                    continue

                # Calculate padded boundaries (relative to the segment)
                extract_start_ms = max(0, int(start_s * 1000) - padding_ms)
                extract_end_ms = min(segment_duration_ms, int(end_s * 1000) + padding_ms)
                
                # Extract with padding
                soundbite_audio = segment_audio[extract_start_ms:extract_end_ms]
                
                # Simple sanitization for filename
                safe_event_type = re.sub(r'[\s\/:]', '_', event_type)

                # Filename: [detected_term]_[base_filename]_[start_sec]_[end_sec].wav
                soundbite_filename = f"[{safe_event_type}]_{base_filename}_{int(start_s)}_{int(end_s)}_{soundbite_index:03d}.wav"
                soundbite_path = os.path.join(output_dir, soundbite_filename)
                
                # Apply fade
                fade_duration = min(50, soundbite_audio.duration_seconds * 1000 / 4) 
                soundbite_audio = soundbite_audio.fade_in(int(fade_duration)).fade_out(int(fade_duration))

                # Export
                soundbite_audio.export(soundbite_path, format="wav")
                soundbite_paths.append(soundbite_path)
                soundbite_index += 1
                logger.debug(f"  -> Extracted soundbite: {soundbite_path}")

            except Exception as e:
                logger.warning(f"Error extracting soundbite for detection {detection}: {e}")
                continue

    logger.info(f"Extracted {len(soundbite_paths)} soundbites for {os.path.basename(segment_audio_path)}.")
    return soundbite_paths

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Setup basic logging to console for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create test audio with specific sounds for call analysis testing
    test_sr = CLAP_SAMPLE_RATE
    test_duration = 30  # 30 seconds
    test_file = "test_call_audio.wav"
    
    # Generate test audio with different tones to simulate phone sounds
    t = np.linspace(0., test_duration, int(test_sr * test_duration))
    audio_data = np.zeros_like(t)
    
    # Add some test tones at different times
    # Dial tone (350Hz + 440Hz)
    dial_tone_start = 5
    dial_tone_duration = 3
    mask = (t >= dial_tone_start) & (t < dial_tone_start + dial_tone_duration)
    audio_data[mask] = 0.3 * (np.sin(2 * np.pi * 350 * t[mask]) + np.sin(2 * np.pi * 440 * t[mask]))
    
    # Busy signal (480Hz + 620Hz)
    busy_start = 15
    busy_duration = 3
    mask = (t >= busy_start) & (t < busy_start + busy_duration)
    audio_data[mask] = 0.3 * (np.sin(2 * np.pi * 480 * t[mask]) + np.sin(2 * np.pi * 620 * t[mask]))
    
    # DTMF tone (example: digit 1 - 697Hz + 1209Hz)
    dtmf_start = 25
    dtmf_duration = 0.5
    mask = (t >= dtmf_start) & (t < dtmf_start + dtmf_duration)
    audio_data[mask] = 0.3 * (np.sin(2 * np.pi * 697 * t[mask]) + np.sin(2 * np.pi * 1209 * t[mask]))
    
    try:
        # Save test audio
        sf.write(test_file, audio_data, test_sr, subtype='PCM_16')
        logging.info(f"Created test audio file: {test_file}")
        
        # Test detection with call analysis settings
        detected_events, cut_segments = detect_and_cut_audio(
            audio_path=test_file,
            output_dir="./test_output",
            enable_cutting=True,
            chunk_duration_s=DEFAULT_CALL_CHUNK_DURATION,
            threshold=DEFAULT_CALL_THRESHOLD,
            target_prompts=CALL_ANALYSIS_PROMPTS
        )
        
        if detected_events:
            print("\nDetected Sound Events:")
            for event in detected_events:
                print(f"- Time: {event['start']:.2f}s - {event['end']:.2f}s")
                print(f"  Label: {event['type']}")
                print(f"  Confidence: {event['confidence']:.3f}")
                
        if cut_segments:
            print("\nCut Segments:")
            for segment in cut_segments:
                print(f"- Segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
                print(f"  Duration: {segment['duration']:.2f}s")
                print(f"  Trigger: {segment['type']}")
                print(f"  Path: {segment['path']}")
                
    except Exception as e:
        logging.error(f"Error during test: {e}", exc_info=True)
    finally:
        # Clean up test files
        if os.path.exists(test_file):
            os.remove(test_file)
            logging.info(f"Removed test audio file: {test_file}") 