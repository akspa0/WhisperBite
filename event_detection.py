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
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        target_events: List[str],
        threshold: float = 0.98
    ) -> List[Dict]:
        """Process a chunk of audio data and detect events."""
        if self.model is None:
            self.load_model()
        
        try:
            # Process audio input
            with torch.amp.autocast(device_type=self.device):
                inputs = self.processor(
                    audios=audio_data,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Process text prompts
                text_inputs = self.processor(
                    text=target_events,
                    return_tensors="pt",
                    padding=True
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    audio_features = self.model.get_audio_features(**inputs)
                    text_features = self.model.get_text_features(**text_inputs)
                    
                    # Calculate similarity scores
                    similarity = torch.nn.functional.cosine_similarity(
                        audio_features[:, None],
                        text_features[None, :],
                        dim=-1
                    )
            
            # Find matches above threshold
            matches = torch.where(similarity > threshold)
            events = []
            
            for audio_idx, text_idx in zip(*matches):
                events.append({
                    'type': target_events[text_idx],
                    'confidence': float(similarity[audio_idx, text_idx])
                })
            
            return events
            
        except Exception as e:
            logging.error(f"Error processing chunk: {str(e)}")
            logging.error("Full traceback:", exc_info=True)
            return []
        finally:
            torch.cuda.empty_cache()
    
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
        try:
            # Load audio file
            logging.info(f"Loading audio file: {audio_path}")
            audio_data, sr = sf.read(audio_path)
            duration = len(audio_data) / sr
            logging.info(f"Audio duration: {duration:.1f}s")
            
            # Resample if needed
            if sr != self.sample_rate:
                import librosa
                logging.info(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            
            # Process in chunks
            chunk_size = int(chunk_duration * self.sample_rate)
            num_chunks = int(np.ceil(len(audio_data) / chunk_size))
            
            all_events: Dict[str, List[Dict]] = {event: [] for event in target_events}
            
            with tqdm(total=num_chunks, desc="Detecting events") as pbar:
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(audio_data))
                    chunk = audio_data[start_idx:end_idx]
                    
                    # Get chunk events
                    chunk_events = self.process_audio_chunk(
                        audio_data=chunk,
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
            return []
        
        # Sort by confidence
        sorted_events = sorted(events, key=lambda x: x['confidence'], reverse=True)
        
        # Keep track of used time windows
        used_times = set()
        filtered_events = []
        
        for event in sorted_events:
            # Get the time window this event falls into
            time_window = int(event['start'])
            
            # Check if any nearby windows are used
            nearby_used = False
            for t in range(
                time_window - int(min_gap_seconds) + 1,
                time_window + int(min_gap_seconds)
            ):
                if t in used_times:
                    nearby_used = True
                    break
            
            if not nearby_used:
                filtered_events.append(event)
                used_times.add(time_window)
        
        # Sort by time for final output
        return sorted(filtered_events, key=lambda x: x['start'])

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