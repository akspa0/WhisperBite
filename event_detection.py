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

class EventDetector:
    def __init__(
        self,
        model_id: str = CLAP_MODEL_ID,
        sample_rate: int = CLAP_SAMPLE_RATE,
        device: Optional[str] = None
    ):
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu') # <<< Use original logic
        
        logging.info(f"Initializing EventDetector with device: {self.device}")
        self.model = None
        self.processor = None
    
    def load_model(self):
        """Load CLAP model and processor."""
        if self.model is None:
            t_start = time.time()
            try:
                self.model = ClapModel.from_pretrained(self.model_id)
                self.processor = ClapProcessor.from_pretrained(self.model_id)
                self.model = self.model.to(self.device)
                self.model.eval()
                logging.info(f"Model loaded in {time.time() - t_start:.1f}s")
            except Exception as e:
                logging.error(f"Failed to load model: {str(e)}")
                raise
    
    def process_audio_chunk(
        self,
        audio_data: np.ndarray,
        text_features: torch.Tensor,
        target_events: List[str],
        threshold: float = 0.98
    ) -> List[Dict]:
        """Process a chunk of audio data and detect events."""
        # Ensure model is loaded (should be done by detect_events now)
        if not self.model or not self.processor:
             logging.error("process_audio_chunk called without model loaded.")
             return []
             
        try:
            # Process audio input
            logging.debug("[CHUNKPROC] Processing audio...")
            t_proc_start = time.time()
            # with torch.amp.autocast(device_type=self.device): # <<< Disable autocast for debugging -> RE-ENABLE
            with torch.amp.autocast(device_type=self.device): # <<< Re-enable autocast
                logging.debug(f"[CHUNKPROC] Entering autocast context: {time.time() - t_proc_start:.4f}s")
                t_inner_start = time.time()
                
                t_call_processor = time.time()
                inputs = self.processor(
                    audios=audio_data,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True
                )
                logging.debug(f"[CHUNKPROC] self.processor() call took: {time.time() - t_call_processor:.4f}s")
                
                t_call_to_device = time.time()
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logging.debug(f"[CHUNKPROC] inputs.to(device) call took: {time.time() - t_call_to_device:.4f}s")
                logging.debug("[CHUNKPROC] Audio inputs processed and moved to device.")
                
                # Get embeddings
                logging.debug("[CHUNKPROC] Getting audio features...")
                with torch.no_grad():
                    logging.debug(f"[CHUNKPROC] Entering no_grad context: {time.time() - t_inner_start:.4f}s")
                    # <<< Log Devices >>>
                    logging.debug(f"[CHUNKPROC] Model device: {next(self.model.parameters()).device}")
                    logging.debug(f"[CHUNKPROC] Audio input device: {inputs['input_features'].device if 'input_features' in inputs else 'N/A'}")
                    # <<< End Log >>>
                    
                    t_call_get_features = time.time()
                    audio_features = self.model.get_audio_features(**inputs)
                    logging.debug(f"[CHUNKPROC] model.get_audio_features() call took: {time.time() - t_call_get_features:.4f}s")
                    logging.debug("[CHUNKPROC] Audio features obtained.")

                    # Calculate similarity scores using precomputed text_features
                    logging.debug("[CHUNKPROC] Calculating similarity...")
                    t_call_similarity = time.time()
                    similarity = torch.nn.functional.cosine_similarity(
                        audio_features[:, None],
                        text_features[None, :],
                        dim=-1
                    )
                    logging.debug(f"[CHUNKPROC] similarity calculation took: {time.time() - t_call_similarity:.4f}s")
                    logging.debug("[CHUNKPROC] Similarity calculated.")
        
            # Find matches above threshold
            matches = torch.where(similarity > threshold)
            events = []
            
            for audio_idx, text_idx in zip(*matches):
                events.append({
                    'type': target_events[text_idx],
                    'confidence': float(similarity[audio_idx, text_idx])
                })
            
            logging.debug(f"[CHUNKPROC] Found {len(events)} events in chunk.")
            return events
            
        except Exception as e:
            logging.error(f"Error processing chunk: {str(e)}")
            logging.error("Full traceback:", exc_info=True)
            return []
        finally:
            # Explicitly delete intermediate tensors if needed, though autocast and loop should handle scope
            del inputs
            del audio_features
            del similarity
            torch.cuda.empty_cache()
            logging.debug("[CHUNKPROC] Chunk processing finished, cache cleared.")
    
    def detect_events(
        self,
        audio_path: str,
        target_events: List[str] = DEFAULT_EVENTS,
        chunk_duration: float = 5.0,
        threshold: float = 0.98,
        min_gap: float = 1.0
    ) -> Dict[str, List[Dict]]:
        """
        Detect sound events in raw audio file.
        
        Args:
            audio_path: Path to audio file
            target_events: List of event types to detect
            chunk_duration: Duration of each chunk in seconds
            threshold: Confidence threshold
            min_gap: Minimum gap between events in seconds
            
        Returns:
            Dictionary mapping event types to lists of detections
        """
        # <<< Load model ONCE before the loop >>>
        self.load_model()
        if not self.model or not self.processor:
            logging.error("Model or processor failed to load. Cannot detect events.")
            return {event: [] for event in target_events}
            
        try:
            # <<< Precompute Text Features >>>
            logging.info("Precomputing text features...")
            t_text_start = time.time()
            try:
                 with torch.amp.autocast(device_type=self.device):
                    text_inputs = self.processor(
                        text=target_events,
                        return_tensors="pt",
                        padding=True
                    )
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    with torch.no_grad():
                        precomputed_text_features = self.model.get_text_features(**text_inputs)
                 logging.info(f"Text features precomputed in {time.time() - t_text_start:.2f}s")
            except Exception as e:
                 logging.error(f"Failed to precompute text features: {e}", exc_info=True)
                 return {event: [] for event in target_events} # Cannot proceed without text features
            # <<< End Precompute >>>

            # Load audio file
            logging.info(f"Loading audio file for event detection: {audio_path}")
            t_load_start = time.time()
            audio_data, sr = sf.read(audio_path)
            logging.info(f"Successfully loaded {audio_path} in {time.time() - t_load_start:.2f}s with sample rate {sr}")
            duration = len(audio_data) / sr
            logging.info(f"Audio duration: {duration:.1f}s")
            
            # --- REMOVE Resampling Block ---
            # Resampling should happen upstream via ffmpeg
            assert sr == self.sample_rate, \
                f"[EventDetector] Expected sample rate {self.sample_rate} but received {sr}. Resample upstream."
            # if sr != self.sample_rate:
            #     import librosa
            #     logging.info(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
            #     t_resample_start = time.time()
            #     audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            #     logging.info(f"Resampling finished in {time.time() - t_resample_start:.2f}s")
            # --- End REMOVE ---
            
            # Process in chunks
            chunk_size = int(chunk_duration * self.sample_rate)
            num_chunks = int(np.ceil(len(audio_data) / chunk_size))
            
            all_events: Dict[str, List[Dict]] = {event: [] for event in target_events}
            
            with tqdm(total=num_chunks, desc="Detecting events") as pbar:
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(audio_data))
                    chunk = audio_data[start_idx:end_idx]
                    
                    # Skip very short final chunks if necessary
                    if len(chunk) < self.sample_rate * 0.1: # Less than 100ms
                        logging.debug(f"Skipping very short final chunk {i}")
                        continue

                    # Get chunk events using precomputed text features
                    chunk_events = self.process_audio_chunk(
                        audio_data=chunk,
                        text_features=precomputed_text_features,
                        target_events=target_events,
                        threshold=threshold
                    )
                    
                    # Add time information to events
                    chunk_start = start_idx / self.sample_rate
                    for event in chunk_events:
                        event['start'] = chunk_start
                        event['end'] = chunk_start + (end_idx - start_idx) / self.sample_rate
                        all_events[event['type']].append(event)
                    
                    pbar.update(1)
                    pbar.set_postfix({'Events': sum(len(e) for e in all_events.values())})
            
            # Apply temporal NMS to each event type
            for event_type in all_events:
                all_events[event_type] = self.apply_temporal_nms(
                    all_events[event_type],
                    min_gap_seconds=min_gap
                )
            
            return all_events
            
        except Exception as e:
            logging.error(f"Error detecting events: {str(e)}")
            logging.error("Full traceback:", exc_info=True)
            return {event: [] for event in target_events}
    
    @staticmethod
    def apply_temporal_nms(
        events: List[Dict],
        min_gap_seconds: float = 1.0
    ) -> List[Dict]:
        """Apply temporal non-maximum suppression to events."""
        if not events:
            logging.info("[NMS] Input events list is empty.")
            return []
        
        # Log input events
        logging.info(f"[NMS] Applying NMS to {len(events)} events with min_gap={min_gap_seconds}s")
        
        # Sort by confidence
        sorted_events = sorted(events, key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        # Keep track of used time windows
        used_times = set()
        filtered_events = []
        
        logging.debug("[NMS] Processing sorted events:")
        for i, event in enumerate(sorted_events):
            start_time = event.get('start')
            confidence = event.get('confidence', 0.0)
            event_type = event.get('type', 'Unknown')
            
            # Ensure start time is valid
            if start_time is None or not isinstance(start_time, (int, float)):
                logging.warning(f"  [NMS Skip {i}] Invalid or missing start time for event: {event}")
                continue
                
            time_window = int(start_time)
            logging.debug(f"  [NMS Check {i}] Event: {event_type}@{start_time:.2f} (Conf: {confidence:.3f}, Window: {time_window})" )
            
            # Check if any nearby windows are used
            nearby_used = False
            for t in range(
                time_window - int(min_gap_seconds) + 1,
                time_window + int(min_gap_seconds)
            ):
                if t in used_times:
                    nearby_used = True
                    logging.debug(f"    -> Nearby window {t} already used. Suppressing.")
                    break
            
            if not nearby_used:
                logging.debug(f"    -> Keeping event. Marking window {time_window} as used.")
                filtered_events.append(event)
                used_times.add(time_window)
        
        # Sort by time for final output
        final_sorted_events = sorted(filtered_events, key=lambda x: x.get('start', 0.0))
        logging.info(f"[NMS] Filtered events count: {len(final_sorted_events)}")
        return final_sorted_events

def detect_and_save_events(
    audio_path: str,
    output_dir: str,
    target_events: List[str] = DEFAULT_EVENTS,
    threshold: float = 0.98,
    min_gap: float = 1.0
) -> Dict[str, List[Dict]]:
    """
    Detect events in audio file and save results.
    
    Args:
        audio_path: Path to audio file
        output_dir: Output directory
        target_events: List of event types to detect
        threshold: Confidence threshold
        min_gap: Minimum gap between events in seconds
        
    Returns:
        Dictionary of detected events by type
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize detector
        detector = EventDetector()
        
        # Detect events
        events = detector.detect_events(
            audio_path=audio_path,
            target_events=target_events,
            threshold=threshold,
            min_gap=min_gap
        )
        
        # Save results
        import json
        output_path = os.path.join(output_dir, "events.json")
        with open(output_path, 'w') as f:
            json.dump(events, f, indent=2)
        
        logging.info(f"Saved event detection results to {output_path}")
        return events
        
    except Exception as e:
        logging.error(f"Error in detect_and_save_events: {str(e)}")
        logging.error("Full traceback:", exc_info=True)
        return {event: [] for event in target_events} 