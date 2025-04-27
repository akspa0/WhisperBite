import os
import logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile # Often needed as a backend for librosa loading/saving

# Constants
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
YAMNET_SAMPLE_RATE = 16000
# TODO: Make threshold configurable?
DETECTION_THRESHOLD = 0.5 
# Define the specific classes we are interested in from the YAMNet class map
# Find these display names in yamnet_class_map.csv
TARGET_SOUND_CLASSES = [
    'Telephone bell ringing',
    'Ringtone',
    'Dial tone',
    'Busy signal',
    # Add more target sounds here if needed, e.g., 'Fire alarm', 'Siren' 
]

# Global variable to hold the loaded model and class map
yamnet_model = None
yamnet_class_map = None

def load_yamnet_model():
    """Loads the YAMNet model from TensorFlow Hub and its class map."""
    global yamnet_model, yamnet_class_map
    if yamnet_model is None:
        try:
            logging.info("Loading YAMNet model from TensorFlow Hub...")
            yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
            logging.info("YAMNet model loaded successfully.")
            
            # Load the class map
            class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
            # Assuming the class map CSV is packaged with the TF Hub model or accessible
            # We might need a more robust way to locate this file if it's not automatic
            try:
                 # TODO: This loading needs refinement based on how TF Hub packages it
                 # For now, let's assume it's loadable directly or we might need to fetch/include it.
                 # A simple approach might be to include the CSV in our repo.
                 # Placeholder: Manually create a dict for now based on search results
                 # We need index -> display_name mapping
                 # Example from yamnet_class_map.csv: 
                 # 459, /m/0160xk, Telephone bell ringing
                 # 460, /m/07p6fty, Ringtone
                 # 462, /m/02mkj, Dial tone
                 # 463, /m/01b_rk, Busy signal
                 yamnet_class_map = { 
                     459: 'Telephone bell ringing',
                     460: 'Ringtone',
                     462: 'Dial tone',
                     463: 'Busy signal',
                     # Add more mappings if TARGET_SOUND_CLASSES expands
                 }
                 logging.info(f"Loaded YAMNet class map (found {len(yamnet_class_map)} classes).")
                 # Verify target classes exist in the map
                 target_indices = get_target_class_indices()
                 if not target_indices:
                     logging.error("Could not find indices for any target sound classes!")
                     yamnet_model = None # Prevent usage if map is wrong
                     return False

            except Exception as e:
                 logging.error(f"Failed to load or parse YAMNet class map from {class_map_path}: {e}")
                 yamnet_model = None # Invalidate model if map fails
                 return False
                 
        except Exception as e:
            logging.error(f"Failed to load YAMNet model from TF Hub: {e}")
            yamnet_model = None
            return False
    return yamnet_model is not None and yamnet_class_map is not None

def get_target_class_indices():
    """Gets the numerical indices for the target sound classes."""
    if not yamnet_class_map:
        logging.error("YAMNet class map not loaded.")
        return {}
        
    target_indices = {}
    # Invert the map for easy lookup: display_name -> index
    name_to_index = {name: index for index, name in yamnet_class_map.items()} 
    
    for target_name in TARGET_SOUND_CLASSES:
        if target_name in name_to_index:
            target_indices[target_name] = name_to_index[target_name]
        else:
            logging.warning(f"Target class '{target_name}' not found in YAMNet class map.")
            
    return target_indices

def detect_sound_events(audio_path):
    """
    Detects specified sound events in an audio file using YAMNet.

    Args:
        audio_path (str): Path to the audio file (e.g., no_vocals.wav).

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              detected sound event segment with keys like 'speaker', 'start', 
              'end', 'text' (containing class name and confidence). Returns 
              an empty list if the model fails to load or no events are detected.
    """
    if not load_yamnet_model():
        logging.error("YAMNet model is not available. Skipping sound detection.")
        return []

    logging.info(f"Starting sound event detection for: {audio_path}")
    detected_events = []
    
    try:
        # Load audio using librosa, ensuring correct sample rate and mono
        waveform, sr = librosa.load(audio_path, sr=YAMNET_SAMPLE_RATE, mono=True)
        logging.info(f"Loaded audio with shape {waveform.shape} at {sr}Hz.")
        
        if waveform.size == 0:
             logging.warning(f"Audio file {audio_path} is empty or could not be loaded correctly.")
             return []

        # --- YAMNet Inference ---
        # Run inference. The model expects float32 [-1.0, 1.0] but hub.load often handles conversion.
        # YAMNet produces scores per frame. Frame hop is typically ~0.48s.
        scores, embeddings, log_mel_spectrogram = yamnet_model(waveform)
        scores = scores.numpy() # Shape: (num_frames, num_classes=521)
        
        # --- Process Scores ---
        target_indices_map = get_target_class_indices() # Map: name -> index
        if not target_indices_map:
             return [] # Exit if we can't map targets

        num_frames, num_classes = scores.shape
        # Frame duration isn't exactly fixed, but hop is ~0.48s, window ~0.96s
        # Let's use the hop time for segment boundaries for simplicity for now.
        frame_hop_seconds = yamnet_model.params.patch_hop_seconds # Accessing model params directly
        frame_start_times = np.arange(num_frames) * frame_hop_seconds

        logging.info(f"YAMNet produced scores for {num_frames} frames.")
        
        sound_event_counter = 0 # For unique sequence if needed later

        for target_name, target_index in target_indices_map.items():
            # Find frames where the target class score exceeds the threshold
            positive_frames = np.where(scores[:, target_index] > DETECTION_THRESHOLD)[0]
            
            if len(positive_frames) > 0:
                logging.info(f"Detected potential '{target_name}' in {len(positive_frames)} frames.")
                
                # --- Simple Merging of Consecutive Frames ---
                # Find contiguous blocks of positive frames
                # Adapted from: https://stackoverflow.com/a/4495197/1167783
                if not positive_frames.any(): continue # Skip if empty after check
                
                merged_events_for_class = []
                start_frame_idx = 0
                while start_frame_idx < len(positive_frames):
                    # Find the end of the contiguous block
                    end_frame_idx = start_frame_idx
                    while (end_frame_idx + 1 < len(positive_frames) and 
                           positive_frames[end_frame_idx + 1] == positive_frames[end_frame_idx] + 1):
                        end_frame_idx += 1
                    
                    # Get segment details
                    start_frame = positive_frames[start_frame_idx]
                    end_frame = positive_frames[end_frame_idx]
                    
                    # Calculate start/end times
                    # Start time is the beginning of the first frame's window
                    # End time is the end of the last frame's window (start + window_duration)
                    segment_start_time = frame_start_times[start_frame]
                    # End time needs the window length (patch_window_seconds)
                    segment_end_time = frame_start_times[end_frame] + yamnet_model.params.patch_window_seconds 

                    # Get average confidence over the segment
                    avg_confidence = np.mean(scores[start_frame : end_frame + 1, target_index])

                    merged_events_for_class.append({
                        'speaker': 'SOUND', 
                        'start': segment_start_time,
                        'end': segment_end_time,
                        # Add confidence to the text label
                        'text': f"{target_name} (Confidence: {avg_confidence:.2f})", 
                        'audio_file': None, 
                        'transcript_file': None,
                        'sequence': sound_event_counter # Increment unique counter
                    })
                    sound_event_counter += 1
                    
                    # Move to the next block
                    start_frame_idx = end_frame_idx + 1
                    
                detected_events.extend(merged_events_for_class)
                logging.info(f"Merged into {len(merged_events_for_class)} segments for '{target_name}'.")

    except Exception as e:
        logging.error(f"Error during sound event detection for {audio_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return [] # Return empty list on error

    # Sort final events by start time before returning
    detected_events.sort(key=lambda x: x['start'])
    logging.info(f"Sound detection finished. Found {len(detected_events)} events in total.")
    
    return detected_events

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Configure logging for standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Create a dummy audio file for testing (e.g., silence with a beep)
    # Requires soundfile library: pip install soundfile
    try:
        import soundfile as sf
        sr_test = YAMNET_SAMPLE_RATE
        duration_test = 5 # seconds
        silence = np.zeros(int(sr_test * duration_test))
        
        # Add a short beep (sine wave) around 2 seconds
        beep_freq = 440 # Hz (A4)
        beep_duration = 0.5 # seconds
        beep_start_sample = int(sr_test * 2)
        t = np.linspace(0., beep_duration, int(sr_test * beep_duration))
        beep_signal = 0.5 * np.sin(2. * np.pi * beep_freq * t)
        
        # Ensure beep signal fits within silence array bounds
        if beep_start_sample + len(beep_signal) <= len(silence):
             silence[beep_start_sample : beep_start_sample + len(beep_signal)] = beep_signal
        
        test_file_path = "test_beep.wav"
        sf.write(test_file_path, silence, sr_test)
        logging.info(f"Created test audio file: {test_file_path}")

        # Add 'Beep, bleep' to target classes for this test run ONLY
        TARGET_SOUND_CLASSES.append('Beep, bleep')
        yamnet_class_map[508] = 'Beep, bleep' # Add mapping for beep
        
        # Run detection
        detected = detect_sound_events(test_file_path)
        print("\n--- Detected Events ---")
        for event in detected:
            print(event)
        print("---------------------")
            
        # Clean up test file
        # os.remove(test_file_path)

    except ImportError:
        logging.warning("Soundfile library not found. Cannot run standalone test.")
    except Exception as e:
        logging.error(f"Error in standalone test: {e}") 