import os
import torch
import scipy.io.wavfile
import numpy as np
from pyannote.audio import Pipeline
from typing import Optional, Dict, Any, Tuple
from core.audio_utils import load_audio
from pydub import AudioSegment

def initialize_diarization_pipeline(hf_token: Optional[str] = None) -> Pipeline:
    """Initialize the Pyannote speaker diarization pipeline with Hugging Face token."""
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token is None:
            raise ValueError(
                "Hugging Face token (HF_TOKEN) not provided. "
                "Set it as an environment variable or pass it directly."
            )
    
    # Initialize the speaker diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    
    return pipeline

def run_diarization_and_segmentation(
    pipeline: Pipeline,
    prepared_audio_path: str,
    output_dir: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> Dict[str, Any]:
    """Run speaker diarization and then segment audio based on speaker turns."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running diarization on: {prepared_audio_path}")
    diarization = pipeline(
        prepared_audio_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    
    # Save RTTM file
    output_rttm_path = os.path.join(output_dir, "diarization.rttm")
    with open(output_rttm_path, "w") as rttm_file:
        diarization.write_rttm(rttm_file)
    print(f"RTTM file saved to: {output_rttm_path}")

    # Load the prepared audio file once for segmentation
    full_audio_waveform, sample_rate = load_audio(prepared_audio_path)
    if sample_rate != 16000:
        print(f"Warning: Prepared audio sample rate is {sample_rate}Hz, expected 16000Hz. Diarization might be affected if pipeline didn't resample.")

    # Create base directory for speaker segments
    speaker_segments_base_dir = os.path.join(output_dir, "speaker_segments")
    os.makedirs(speaker_segments_base_dir, exist_ok=True)

    speaker_audio_data = {}
    segments_manifest = []
    
    print("Segmenting audio by speaker...")
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        speaker_dir = os.path.join(speaker_segments_base_dir, speaker_label)
        os.makedirs(speaker_dir, exist_ok=True)
        
        turn_idx = sum(1 for s in segments_manifest if s["speaker_id"] == speaker_label)
        segment_filename = f"{speaker_label}_turn_{turn_idx}.wav"
        segment_output_path = os.path.join(speaker_dir, segment_filename)
        
        start_sample = int(turn.start * sample_rate)
        end_sample = int(turn.end * sample_rate)
        turn_waveform = full_audio_waveform[start_sample:end_sample]
        
        scipy.io.wavfile.write(segment_output_path, sample_rate, turn_waveform.astype(np.float32))
        
        relative_segment_path = os.path.relpath(segment_output_path, output_dir)
        speaker_audio_data.setdefault(speaker_label, []).append(relative_segment_path)

        segments_manifest.append({
            "speaker_id": speaker_label,
            "start_time": round(turn.start, 3),
            "end_time": round(turn.end, 3),
            "duration": round(turn.duration, 3),
            "file_path": relative_segment_path
        })
        print(f"Saved segment: {segment_output_path}")

    # Save segment manifest
    segment_manifest_path = os.path.join(speaker_segments_base_dir, "speaker_segments_manifest.json")
    import json
    with open(segment_manifest_path, "w") as f:
        json.dump(segments_manifest, f, indent=2)
    print(f"Speaker segments manifest saved to: {segment_manifest_path}")
    
    # Return metadata
    return {
        "identified_speakers": list(diarization.labels()),
        "total_speech_duration": round(diarization.get_timeline().support().duration(), 3),
        "rttm_file_path": os.path.relpath(output_rttm_path, output_dir),
        "speaker_segments_manifest": os.path.relpath(segment_manifest_path, output_dir),
    }

# Remove old compatibility assignments if they are no longer relevant
# initialize_diarization_pipeline_old_ref = initialize_separation_pipeline # Example if needed
# run_diarization_old_ref = run_separation # Example if needed

# To maintain potential backward compatibility for now if other (unseen) parts of the codebase
# use these exact names, we can assign the new functions to the old names.
# However, it's better to update call sites. For now, let's comment out the old ones.
# initialize_diarization_pipeline_old_ref = initialize_separation_pipeline # Example if needed
# run_diarization_old_ref = run_separation # Example if needed 