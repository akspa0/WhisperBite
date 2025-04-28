import os
import logging
import torch
import librosa
import numpy as np
from transformers import ClapModel, ClapProcessor

# Constants
# Model chosen based on general availability and performance. Others could be tested.
CLAP_MODEL_ID = "laion/clap-htsat-unfused" 
# Typical sample rate for CLAP models
CLAP_SAMPLE_RATE = 48000 
# TODO: Make these configurable via args/UI
DEFAULT_CHUNK_DURATION_S = 5.0 # Process audio in 5-second chunks
DEFAULT_DETECTION_THRESHOLD = 0.7 # Confidence threshold for detection

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

    logging.info(f"Starting CLAP sound event detection for: {audio_path}")
    logging.info(f"Parameters: Chunk Duration={chunk_duration_s}s, Threshold={threshold}")
    
    detected_events = []
    prompts_to_use = target_prompts if target_prompts else TARGET_SOUND_PROMPTS
    logging.info(f"Using target prompts: {prompts_to_use}")

    try:
        # Load audio using librosa, ensuring correct sample rate
        # Load as mono for simplicity, as CLAP typically processes single channel audio embeddings effectively.
        logging.info(f"Loading audio: {audio_path}...")
        waveform, sr = librosa.load(audio_path, sr=CLAP_SAMPLE_RATE, mono=True)
        logging.info(f"Loaded audio with shape {waveform.shape} at {sr}Hz ({librosa.get_duration(y=waveform, sr=sr):.2f} seconds).")
        
        if waveform.size == 0:
             logging.warning(f"Audio file {audio_path} is empty or could not be loaded correctly.")
             return []

        num_samples = len(waveform)
        chunk_size = int(chunk_duration_s * CLAP_SAMPLE_RATE)
        num_chunks = int(np.ceil(num_samples / chunk_size))
        
        logging.info(f"Processing audio in {num_chunks} chunks of ~{chunk_duration_s}s...")

        sound_event_counter = 0 # For unique sequence if needed later

        for i in range(num_chunks):
            start_sample = i * chunk_size
            end_sample = min((i + 1) * chunk_size, num_samples)
            chunk_waveform = waveform[start_sample:end_sample]
            
            # Calculate timestamps for this chunk
            start_time = start_sample / CLAP_SAMPLE_RATE
            end_time = end_sample / CLAP_SAMPLE_RATE
            
            if len(chunk_waveform) == 0:
                continue

            try:
                 # Process with CLAP
                inputs = clap_processor(
                    text=prompts_to_use, 
                    audios=[chunk_waveform], # Processor expects a list of audio arrays
                    return_tensors="pt", 
                    padding=True, 
                    sampling_rate=CLAP_SAMPLE_RATE
                ).to(device)

                with torch.no_grad():
                    outputs = clap_model(**inputs)
                
                # logits_per_audio: [batch_size, num_prompts] 
                # In our case batch_size is 1 (one chunk)
                logits_per_audio = outputs.logits_per_audio 
                
                # Convert logits to probabilities (0-1 range)
                probs = torch.sigmoid(logits_per_audio).squeeze().cpu().numpy() # Squeeze to remove batch dim, move to cpu

                # Check against threshold for each prompt
                for prompt_idx, prompt_text in enumerate(prompts_to_use):
                    confidence = probs[prompt_idx]
                    if confidence >= threshold:
                        logging.debug(f"Detected '{prompt_text}' (Conf: {confidence:.3f}) at {start_time:.2f}s - {end_time:.2f}s")
                        detected_events.append({
                            'speaker': 'SOUND', # Keep consistent label for sound events
                            'start': start_time,
                            'end': end_time,
                            'text': f"{prompt_text}", # Just the label, confidence is separate now
                            'confidence': float(confidence), # Store confidence
                            'audio_file': None, # Keep structure consistent if needed elsewhere
                            'transcript_file': None,
                            'sequence': sound_event_counter # Maintain sequence if helpful
                        })
                        sound_event_counter += 1
            
            except Exception as chunk_e:
                logging.error(f"Error processing chunk {i} ({start_time:.2f}s-{end_time:.2f}s): {chunk_e}", exc_info=True)
                continue # Skip to next chunk on error

    except Exception as e:
        logging.error(f"Failed during sound detection for {audio_path}: {e}", exc_info=True)
        return [] # Return empty list on major failure

    logging.info(f"Finished CLAP sound detection. Found {len(detected_events)} potential events.")
    
    # Optional: Add merging logic for consecutive identical events here if needed
    # For now, returning raw chunk-level detections. Merging can be complex.

    return detected_events


# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Setup basic logging to console for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a dummy audio file for testing (e.g., 15 seconds of sine wave)
    test_sr = CLAP_SAMPLE_RATE
    test_duration = 15 
    test_freq = 440 # A4 note
    t = np.linspace(0., test_duration, int(test_sr * test_duration))
    amplitude = np.iinfo(np.int16).max * 0.5 
    data = amplitude * np.sin(2. * np.pi * test_freq * t)
    
    # Add some silence and maybe another tone to simulate changes
    data[int(5*test_sr):int(7*test_sr)] = 0 # Silence from 5s to 7s
    test_freq2 = 880
    t2 = np.linspace(0., 3, int(test_sr * 3))
    data[int(10*test_sr):int(13*test_sr)] = amplitude * np.sin(2. * np.pi * test_freq2 * t2) # Tone from 10s to 13s


    test_file = "test_clap_audio.wav"
    
    import soundfile as sf # Use soundfile directly for saving test audio
    try:
        sf.write(test_file, data, test_sr, subtype='PCM_16') # Save as 16-bit PCM WAV
        logging.info(f"Created dummy audio file: {test_file}")

        # --- Run Detection ---
        # Use prompts that are unlikely to match the sine wave for testing thresholding
        # And maybe 'speech' or 'music' to see if it gets triggered spuriously
        test_prompts = ["dog barking", "speech", "music", "siren"]
        detected = detect_sound_events(test_file, chunk_duration_s=2.0, threshold=0.5, target_prompts=test_prompts)

        if detected:
            print("Detected Sound Events:")
            for event in detected:
                print(f"- Time: {event['start']:.2f}s - {event['end']:.2f}s, Label: {event['text']}, Confidence: {event['confidence']:.3f}")
        else:
            print("No sound events detected above the threshold.")
            
    except Exception as main_e:
        logging.error(f"Error during test execution: {main_e}", exc_info=True)
    finally:
        # Clean up the dummy file
        if os.path.exists(test_file):
            os.remove(test_file)
            logging.info(f"Removed dummy audio file: {test_file}") 