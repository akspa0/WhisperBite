# core/clap_annotator.py
import os
import torch
import numpy as np
import json
import librosa
from typing import List, Optional, Dict, Any
from transformers import ClapModel, ClapProcessor, AutoProcessor
from pydub import AudioSegment

def chunk_audio(audio_path: str, chunk_size_ms: int = 10000, hop_size_ms: int = 5000) -> List[Dict[str, Any]]:
    """Split audio into overlapping chunks for CLAP processing.
    
    Args:
        audio_path: Path to audio file
        chunk_size_ms: Chunk size in milliseconds
        hop_size_ms: Hop size in milliseconds
    
    Returns:
        List of dictionaries with chunk audio arrays and metadata
    """
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    
    # Get audio duration
    duration_ms = len(audio)
    
    # Create chunks
    chunks = []
    
    for start_ms in range(0, duration_ms, hop_size_ms):
        end_ms = min(start_ms + chunk_size_ms, duration_ms)
        
        # Extract chunk
        chunk = audio[start_ms:end_ms]
        
        # Convert to numpy array
        samples = np.array(chunk.get_array_of_samples())
        
        # Convert to float32 and normalize
        if chunk.sample_width == 2:  # 16-bit audio
            samples = samples.astype(np.float32) / 32768.0
        else:  # Assume 32-bit audio
            samples = samples.astype(np.float32) / 2147483648.0
        
        # If stereo, convert to mono
        if chunk.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        
        # Resample to 48kHz if needed (CLAP expects 48kHz)
        if chunk.frame_rate != 48000:
            samples = librosa.resample(
                samples, 
                orig_sr=chunk.frame_rate, 
                target_sr=48000
            )
        
        chunks.append({
            "audio": samples,
            "start_time": start_ms / 1000.0,  # Convert to seconds
            "end_time": end_ms / 1000.0,
            "sample_rate": 48000
        })
    
    return chunks

def initialize_clap_model(model_name: str = "microsoft/clap-htsat-unfused", device: str = "cuda") -> tuple:
    """Initialize CLAP model and processor."""
    # Check if CUDA is available when device is set to "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Use AutoProcessor for compatibility with both fused and unfused models
    processor = AutoProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).to(device)
    
    return model, processor, device

def detect_events_in_chunks(
    model: ClapModel,
    processor: AutoProcessor,
    audio_chunks: List[Dict[str, Any]],
    prompts: List[str],
    threshold: float = 0.5,
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """Detect events in chunked audio using CLAP prompts."""
    results = []
    
    for chunk in audio_chunks:
        # Process audio input for this chunk
        inputs = processor(
            text=prompts,
            audios=chunk["audio"],
            sampling_rate=chunk["sample_rate"],
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Get scores from model
        with torch.no_grad():
            outputs = model(**inputs)
            # For fused model, logits_per_audio has the scores
            scores = outputs.logits_per_audio
            # Convert to probabilities
            probs = torch.softmax(scores, dim=-1).cpu().numpy()[0]
        
        # Filter by threshold
        for i, (prompt, prob) in enumerate(zip(prompts, probs)):
            if prob >= threshold:
                results.append({
                    "label": prompt,
                    "prompt_used": prompt,
                    "confidence": float(prob),
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"]
                })
    
    # Merge overlapping detections of the same class
    merged_results = []
    current_events = {}
    
    # Group by label
    for event in sorted(results, key=lambda x: (x["label"], x["start_time"])):
        label = event["label"]
        
        if label not in current_events:
            current_events[label] = event
        else:
            current = current_events[label]
            
            # If overlapping, merge
            if event["start_time"] <= current["end_time"]:
                current["end_time"] = max(current["end_time"], event["end_time"])
                current["confidence"] = max(current["confidence"], event["confidence"])
            else:
                # No overlap, add the current event and start a new one
                merged_results.append(current)
                current_events[label] = event
    
    # Add remaining events
    merged_results.extend(current_events.values())
    
    return merged_results

def annotate_audio(
    audio_path: str,
    output_dir: str,
    event_prompts: Optional[List[str]] = None,
    event_threshold: float = 0.5,
    sound_prompts: Optional[List[str]] = None,
    sound_threshold: float = 0.3,
    model_name: str = "microsoft/clap-htsat-unfused",
    device: str = "cuda",
    chunk_size_ms: int = 10000,
    hop_size_ms: int = 5000
) -> Dict[str, Any]:
    """Process audio with CLAP for sound event detection."""
    results = {}
    
    # Skip if no prompts provided
    if not event_prompts and not sound_prompts:
        print("No prompts provided for CLAP analysis, skipping.")
        return results
    
    # Initialize model
    model, processor, actual_device = initialize_clap_model(model_name, device)
    
    # Chunk audio for processing
    print(f"Chunking audio into {chunk_size_ms}ms segments with {hop_size_ms}ms hop size...")
    audio_chunks = chunk_audio(audio_path, chunk_size_ms, hop_size_ms)
    print(f"Created {len(audio_chunks)} chunks for processing")
    
    # Process event prompts
    if event_prompts:
        print(f"Processing {len(event_prompts)} event prompts...")
        events = detect_events_in_chunks(
            model, processor, audio_chunks, 
            event_prompts, event_threshold, actual_device
        )
        results["events"] = events
        
        # Save event results
        os.makedirs(output_dir, exist_ok=True)
        event_path = os.path.join(output_dir, "clap_events.json")
        with open(event_path, "w") as f:
            json.dump(events, f, indent=2)
    
    # Process sound prompts
    if sound_prompts:
        print(f"Processing {len(sound_prompts)} sound prompts...")
        sounds = detect_events_in_chunks(
            model, processor, audio_chunks, 
            sound_prompts, sound_threshold, actual_device
        )
        results["sounds"] = sounds
        
        # Save sound results
        os.makedirs(output_dir, exist_ok=True)
        sound_path = os.path.join(output_dir, "clap_sounds.json")
        with open(sound_path, "w") as f:
            json.dump(sounds, f, indent=2)
    
    # Save combined results
    combined_path = os.path.join(output_dir, "clap_annotations.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results