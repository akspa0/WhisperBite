# core/transcription.py
import os
import torch
import numpy as np
import json
from typing import List, Optional, Dict, Any
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# Ensure soundfile or librosa is available for loading audio segments if not relying on ffmpeg for everything
# For consistency with how diarization.py loads the main prepared audio (using core.audio_utils.load_audio),
# it might be better to have a similar robust way to load segments here, or ensure segments are saved in a very standard WAV format.
# load_audio from audio_utils uses ffmpeg and returns float32 numpy array and sample rate.

# Default model, can be overridden by CLI
DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3"

def initialize_whisper_model(
    model_name: str = DEFAULT_WHISPER_MODEL, 
    device: Optional[str] = None,
    hf_token: Optional[str] = None # For gated models, if any are used in future
) -> tuple:
    """Initialize Whisper model and processor from Hugging Face.
    
    Args:
        model_name: Hugging Face model identifier (e.g., "openai/whisper-large-v3").
        device: Device to load the model on ("cuda", "cpu", or None for auto-detect).
                 If None, uses cuda if available, else cpu.
        hf_token: Hugging Face token for gated models.

    Returns:
        A tuple (model, processor, device_str)
    """
    if device is None:
        computed_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        computed_device = device
    
    print(f"Initializing Whisper model '{model_name}' on device '{computed_device}'...")
    
    try:
        processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if computed_device == "cuda" else torch.float32, # float16 for GPU for speed/memory
            low_cpu_mem_usage=True, # Good for large models
            use_safetensors=True,
            token=hf_token
        )
        model.to(computed_device)
        
        # For generation, especially with word timestamps, creating a pipeline can be easier.
        # However, direct model.generate() also works. Using pipeline for convenience here.
        # Note: The `pipeline` from transformers can simplify chunking and batching later if needed.
        transcribe_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=computed_device,
            torch_dtype=torch.float16 if computed_device == "cuda" else torch.float32,
        )
        print(f"Whisper model '{model_name}' initialized successfully.")
        return transcribe_pipeline, processor, computed_device # Returning pipeline, processor, and device string
    except Exception as e:
        print(f"Error initializing Whisper model: {e}")
        raise

def transcribe_segment(
    transcribe_pipeline: Any, # Expects the ASR pipeline from initialize_whisper_model
    audio_segment_path: str,
    language: Optional[str] = None,
    task: str = "transcribe", # "transcribe" or "translate"
    return_word_timestamps: bool = True
) -> Dict[str, Any]:
    """Transcribe a single audio segment using the initialized Whisper pipeline.

    Args:
        transcribe_pipeline: The ASR pipeline instance.
        audio_segment_path: Path to the audio segment file (e.g., WAV).
        language: Language code for transcription (e.g., "en"). Whisper auto-detects if None.
        task: "transcribe" or "translate".
        return_word_timestamps: Whether to request word-level timestamps.

    Returns:
        A dictionary containing the transcription text and word/chunk timestamps.
        Example: {"text": "Hello world", "chunks": [{"text": "Hello", "timestamp": (0.0, 1.0)}, ...]}
    """
    if not os.path.exists(audio_segment_path):
        raise FileNotFoundError(f"Audio segment not found: {audio_segment_path}")

    print(f"Transcribing segment: {audio_segment_path}...")
    
    # The pipeline handles loading the audio file directly.
    # It also handles resampling to 16kHz mono if needed, which is what Whisper expects.
    # For word timestamps, the pipeline needs generate_kwargs.
    generate_kwargs = {
        "task": task,
        "language": language if language else None, # Important to pass None, not empty string, for auto-detect
        "return_timestamps": "word" if return_word_timestamps else True # 'word' for word-level, True for segment-level
    }

    try:
        # The pipeline can take a path to an audio file or a raw waveform.
        # It returns a dictionary including "text" and potentially "chunks" with timestamps.
        output = transcribe_pipeline(audio_segment_path, generate_kwargs=generate_kwargs)
        
        # Structure of output can vary slightly based on transformers version and parameters.
        # We primarily need "text" and "chunks" (which contain word-level timestamps if requested).
        # If "chunks" is not present or empty, but "text" is, we can use the full text with segment-level timestamps (if available)
        # or just the text.
        
        transcription_result = {
            "text": output.get("text", "").strip(),
            "language_detected": output.get("language", None) # Some pipelines might return this
        }

        if return_word_timestamps and "chunks" in output and output["chunks"]:
            transcription_result["chunks"] = [
                {
                    "text": chunk["text"].strip(), 
                    "timestamp": chunk["timestamp"] # (start_seconds, end_seconds)
                } 
                for chunk in output["chunks"]
            ]
        elif "text" in output: # Fallback if no chunks or word timestamps not requested
             # If `return_timestamps=True` (but not "word"), there might be segment-level timestamps.
             # For now, just capture the full text. The CLI will reconstruct global times from segment start/end.
            pass 

        print(f"Transcription complete for: {audio_segment_path}")
        return transcription_result
    except Exception as e:
        print(f"Error during transcription of {audio_segment_path}: {e}")
        # It might be desirable to return a partial result or error structure here
        # instead of raising, so one failed segment doesn't stop a batch.
        # For now, re-raising to make errors visible.
        raise

# Example helper to save (can be expanded or moved to CLI logic)
def save_transcription_files(
    transcription_result: Dict[str, Any],
    base_output_path_no_ext: str, # e.g., "output_dir/transcripts/speaker_segments/SPEAKER_00/SPEAKER_00_turn_0"
    global_start_time: float = 0.0
):
    """Saves transcription to .txt and .json files.
    Adjusts timestamps in JSON to be global if global_start_time is provided.
    """
    os.makedirs(os.path.dirname(base_output_path_no_ext), exist_ok=True)

    # Save .txt file (just the plain text)
    with open(f"{base_output_path_no_ext}.txt", "w", encoding="utf-8") as f:
        f.write(transcription_result.get("text", ""))

    # Prepare JSON data (potentially with adjusted global timestamps)
    json_output = {
        "text": transcription_result.get("text", ""),
        "language_detected": transcription_result.get("language_detected")
    }
    if "chunks" in transcription_result:
        json_output["chunks"] = [
            {
                "text": chunk["text"],
                "timestamp": (
                    round(chunk["timestamp"][0] + global_start_time, 3),
                    round(chunk["timestamp"][1] + global_start_time, 3)
                )
            }
            for chunk in transcription_result["chunks"]
        ]
    
    # Save .json file
    with open(f"{base_output_path_no_ext}.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"Transcription saved to {base_output_path_no_ext}.txt and .json")

if __name__ == '__main__':
    # Quick test (requires a valid audio file and HF token if model is gated or first download)
    # This is just for basic module testing, full testing via CLI.
    print("Testing transcription module...")
    try:
        # Ensure you have an audio file path here for testing
        # test_audio_path = "/path/to/your/test_segment.wav"
        # example_output_dir = "test_transcribe_out"
        # os.makedirs(example_output_dir, exist_ok=True)

        # if not os.path.exists(test_audio_path):
        #     print(f"Test audio file not found: {test_audio_path}. Skipping interactive test.")
        # else:
        #     print("Initializing model (this may take time on first run)...")
        #     # Set HF_TOKEN environment variable if needed, or pass here
        #     # test_hf_token = os.getenv("HF_TOKEN") 
        #     asr_pipeline, proc, dev = initialize_whisper_model(device="cpu") # Test on CPU to be safe
            
        #     print(f"Transcribing {test_audio_path}...")
        #     result = transcribe_segment(asr_pipeline, test_audio_path, return_word_timestamps=True)
            
        #     print("\nTranscription Result:")
        #     print(f"  Text: {result.get('text')}")
        #     if result.get("language_detected"):
        #          print(f"  Language: {result.get('language_detected')}")
        #     if 'chunks' in result:
        #         print("  Word Timestamps:")
        #         for chunk in result['chunks']:
        #             print(f"    {chunk['timestamp']}: {chunk['text']}")
            
        #     base_name = os.path.join(example_output_dir, "test_output_segment")
        #     save_transcription_files(result, base_name, global_start_time=0.0) # Example save
        #     print(f"\nTest complete. Check files in '{example_output_dir}'")
        print("Module structure defined. Add a test audio file and uncomment main section to run a test.")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc() 