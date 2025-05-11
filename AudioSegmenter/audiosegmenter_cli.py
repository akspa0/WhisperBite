# audiosegmenter_cli.py
import typer
from typing import List, Optional
import os
import sys
import subprocess
import json
from pathlib import Path
from core.audio_utils import prepare_audio
from core.diarization import initialize_diarization_pipeline, run_diarization_and_segmentation
from core.clap_annotator import annotate_audio
from core.transcription import initialize_whisper_model, transcribe_segment, save_transcription_files, DEFAULT_WHISPER_MODEL

app = typer.Typer(help="AudioSegmenter: Speaker diarization, transcription, and sound annotation tool")

def check_system_dependencies():
    """Verify system dependencies (FFmpeg)"""
    try:
        # Try using subprocess to check if ffmpeg is available
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True,
                               check=False)
        
        if result.returncode == 0:
            # FFmpeg is installed and working
            return True
        else:
            raise RuntimeError("FFmpeg command failed")
            
    except (FileNotFoundError, RuntimeError):
        typer.secho(
            "Error: FFmpeg not found! Please install it:\n"
            "  Linux/WSL: sudo apt install ffmpeg\n"
            "  Windows: Download from ffmpeg.org\n"
            "  macOS: brew install ffmpeg", 
            fg=typer.colors.RED
        )
        sys.exit(1)

def normalize_path(path_str: str) -> str:
    """Normalize a path string to ensure correct separators and absolute paths.
    Handles both Windows and Unix-style paths in WSL environment."""
    
    # Debug
    original_path = path_str
    
    # Handle Windows backslash paths in WSL
    if '\\' in path_str:
        # Convert to proper Windows path first, then to WSL path
        # Example: ..\..\test_audio\file.mp3 → /mnt/c/WhisperBite/v7/test_audio/file.mp3
        
        # Replace backslashes with forward slashes
        path_str = path_str.replace('\\', '/')
        
        # Check if it's a relative path
        if path_str.startswith('../') or path_str.startswith('./'):
            # Handle relative paths by first making them absolute, then converting
            base_dir = os.getcwd()
            path = os.path.normpath(os.path.join(base_dir, path_str))
        else:
            path = path_str
    
    # Normalize path
    path = Path(path_str).expanduser().resolve()
    
    # Convert to string
    normalized_path = str(path)
    
    # Debug
    if original_path != normalized_path:
        print(f"Normalized path: {original_path} → {normalized_path}")
    
    return normalized_path

@app.command()
def process(
    input_path: str = typer.Argument(..., help="Path to input audio/video file"),
    output_dir: str = typer.Argument(..., help="Output directory for results"),
    # Audio preprocessing
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Enable/disable audio normalization."),
    target_lufs: float = typer.Option(-14.0, help="Target LUFS level for normalization (e.g., -14.0 for YouTube)"),
    force_resample: bool = typer.Option(True, "--resample/--no-resample", help="Force/disable resample to 16kHz mono."),
    # Diarization
    hf_token: str = typer.Option(None, help="Hugging Face token (default: HF_TOKEN env var). Used for Pyannote and potentially gated HF models."),
    num_speakers: Optional[int] = typer.Option(None, help="Exact speaker count for diarization."),
    min_speakers: int = typer.Option(1, help="Minimum speaker count for diarization."),
    max_speakers: int = typer.Option(5, help="Maximum speaker count for diarization."),
    # NEW: Transcription options
    transcribe: bool = typer.Option(True, "--transcribe/--no-transcribe", help="Enable/disable speech transcription."),
    whisper_model_name: str = typer.Option(DEFAULT_WHISPER_MODEL, help=f"Whisper model name from Hugging Face. Default: {DEFAULT_WHISPER_MODEL}"),
    transcription_language: Optional[str] = typer.Option(None, help="Language code for transcription (e.g., en, es). Whisper auto-detects if None."),
    word_timestamps: bool = typer.Option(True, "--word-timestamps/--no-word-timestamps", help="Enable/disable word-level timestamps in transcription."),
    # CLAP annotation
    clap_model: str = typer.Option("microsoft/clap-htsat-unfused", help="CLAP model name for sound event annotation."),
    device: str = typer.Option("cuda", help="Inference device (e.g., cuda, cpu, mps). Auto-detects if 'cuda' selected but unavailable."),
    event_prompts: Optional[List[str]] = typer.Option(None, help="Event prompts for CLAP (comma-separated strings)."),
    event_threshold: float = typer.Option(0.5, help="Event detection threshold for CLAP."),
    sound_prompts: Optional[List[str]] = typer.Option(None, help="Sound prompts for CLAP (comma-separated strings)."), 
    sound_threshold: float = typer.Option(0.3, help="Sound detection threshold for CLAP."),
    # Debug
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output.")
):
    """Process an audio file: preprocess, diarize, segment, transcribe, and annotate sound events."""
    # Check dependencies
    check_system_dependencies()
    
    # Print original paths for debugging
    print(f"Original input path: {input_path}")
    print(f"Original output directory: {output_dir}")
    
    # Normalize paths to ensure correct format
    input_path = normalize_path(input_path)
    output_dir = normalize_path(output_dir)
    
    print(f"Normalized input path: {input_path}")
    print(f"Normalized output directory: {output_dir}")
    
    if verbose:
        typer.echo(f"🔍 Input file: {input_path}")
        typer.echo(f"📁 Output directory: {output_dir}")
    
    # Validate the input file exists
    if not os.path.isfile(input_path):
        typer.secho(f"Error: Input file not found: {input_path}", fg=typer.colors.RED)
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Audio preprocessing
    if verbose:
        typer.echo("⏳ Preprocessing audio...")
    audio_path = prepare_audio(
        input_path=input_path,
        output_dir=output_dir,
        normalize=normalize,
        target_lufs=target_lufs,
        force_resample=force_resample
    )
    
    # 2. Speaker diarization and segmentation
    if verbose:
        typer.echo("🔊 Running speaker diarization and segmentation...")
    pipeline = initialize_diarization_pipeline(hf_token)
    diarization_results = run_diarization_and_segmentation(
        pipeline=pipeline,
        prepared_audio_path=audio_path,
        output_dir=output_dir,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    
    if verbose:
        typer.echo(f"✅ Identified {len(diarization_results['identified_speakers'])} speakers:")
        for speaker in diarization_results['identified_speakers']:
            typer.echo(f"  - {speaker}")
        typer.echo(f"  RTTM file: {diarization_results['rttm_file_path']}")
        typer.echo(f"  Segments manifest: {diarization_results['speaker_segments_manifest']}")
    
    # 3. NEW: Speech Transcription
    full_transcript_data = []
    if transcribe:
        if verbose: typer.echo("📝 Running speech transcription...")
        
        # Determine device for Whisper (can be different from CLAP if needed, but CLI uses one `device` param)
        # initialize_whisper_model will auto-select cuda if available and device is "cuda"
        asr_pipeline, _, whisper_device = initialize_whisper_model(model_name=whisper_model_name, device=device, hf_token=hf_token)
        if verbose: typer.echo(f"Whisper model initialized on device: {whisper_device}")

        speaker_segments_manifest_path = output_dir / diarization_results['speaker_segments_manifest']
        if not speaker_segments_manifest_path.exists():
            typer.secho(f"Error: Speaker segments manifest not found: {speaker_segments_manifest_path}", fg=typer.colors.RED)
            # Decide if to exit or just skip transcription
        else:
            with open(speaker_segments_manifest_path, 'r', encoding='utf-8') as f:
                speaker_segments = json.load(f)
            
            transcripts_base_dir = output_dir / "transcripts"
            transcripts_segments_dir = transcripts_base_dir / "speaker_segments"
            transcripts_segments_dir.mkdir(parents=True, exist_ok=True)

            for segment_info in speaker_segments:
                segment_wav_path = output_dir / segment_info["file_path"]
                if not segment_wav_path.exists():
                    if verbose: typer.secho(f"Warning: Segment WAV file not found: {segment_wav_path}, skipping transcription.", fg=typer.colors.YELLOW)
                    continue
                
                if verbose: typer.echo(f"Transcribing segment: {segment_wav_path} ({segment_info['speaker_id']} {segment_info['start_time']}-{segment_info['end_time']}) ...")
                
                transcription_result = transcribe_segment(
                    transcribe_pipeline=asr_pipeline,
                    audio_segment_path=str(segment_wav_path),
                    language=transcription_language,
                    return_word_timestamps=word_timestamps
                )
                
                # Determine base path for saving this segment's transcript
                # e.g. .../transcripts/speaker_segments/SPEAKER_00/SPEAKER_00_turn_0 (no ext)
                relative_segment_file_path = Path(segment_info["file_path"])
                # transcripts_segments_dir / SPEAKER_XX / SPEAKER_XX_turn_N (no ext)
                transcript_base_save_path = transcripts_segments_dir / relative_segment_file_path.parent.name / relative_segment_file_path.stem
                
                save_transcription_files(
                    transcription_result=transcription_result,
                    base_output_path_no_ext=str(transcript_base_save_path),
                    global_start_time=segment_info["start_time"] # Pass global start time for timestamp adjustment
                )
                
                # Store for full transcript reconstruction
                # Ensure text and chunks are present before trying to access them
                text = transcription_result.get("text", "").strip() # Ensure text is also stripped here
                word_timestamps_for_full_transcript = [] # Renamed for clarity

                if "chunks" in transcription_result and transcription_result["chunks"]:
                    word_timestamps_for_full_transcript = [
                        {
                            "text": chunk["text"],
                            "timestamp": ( # Adjust to global time here
                                round(chunk["timestamp"][0] + segment_info["start_time"], 3),
                                round(chunk["timestamp"][1] + segment_info["start_time"], 3)
                            )
                        }
                        for chunk in transcription_result["chunks"]
                    ]
                elif text: # If no word timestamps, create a single chunk for the segment using its global times
                    word_timestamps_for_full_transcript = [{
                        "text": text,
                        "timestamp": (round(segment_info["start_time"],3), round(segment_info["end_time"],3))
                    }]

                full_transcript_data.append({
                    "speaker_id": segment_info["speaker_id"],
                    "start_time": round(segment_info["start_time"], 3),
                    "end_time": round(segment_info["end_time"], 3),
                    "text": text,
                    "language_detected": transcription_result.get("language_detected"),
                    # Simpler check: if word_timestamps_for_full_transcript is not empty and its content isn't just the full text
                    "word_timestamps_available": bool(word_timestamps_for_full_transcript) and \
                                               not (len(word_timestamps_for_full_transcript) == 1 and word_timestamps_for_full_transcript[0]["text"] == text),
                    "chunks": word_timestamps_for_full_transcript # Use the globally adjusted chunks
                })
            
            # Sort by start time and save full transcript
            if full_transcript_data:
                full_transcript_data.sort(key=lambda x: x["start_time"])
                full_json_path = transcripts_base_dir / "full_transcript.json"
                with open(full_json_path, 'w', encoding='utf-8') as f:
                    json.dump(full_transcript_data, f, indent=2, ensure_ascii=False)
                if verbose: typer.echo(f"Full transcript JSON saved to: {full_json_path}")
                
                full_txt_path = transcripts_base_dir / "full_transcript.txt"
                with open(full_txt_path, 'w', encoding='utf-8') as f:
                    for item in full_transcript_data:
                        start_hms = str(Path(output_dir) / item["start_time"]) # Poor man's sec_to_hms, replace later
                        end_hms = str(Path(output_dir) / item["end_time"]) # Poor man's sec_to_hms, replace later
                        # TODO: Convert seconds to HH:MM:SS.mmm format for TXT file
                        f.write(f"[{item['start_time']:.3f}s --> {item['end_time']:.3f}s] {item['speaker_id']}: {item['text']}\n")
                if verbose: typer.echo(f"Full transcript TXT saved to: {full_txt_path}")
            else:
                if verbose: typer.echo("No segments were transcribed to create a full transcript.")
    else:
        if verbose: typer.echo("Transcription disabled via --no-transcribe.")

    # 4. Sound annotation (if prompts provided)
    if event_prompts or sound_prompts:
        if verbose: typer.echo("🎵 Running sound annotation...")
        # Note: CLAP annotation should use prepared_audio_path_str for consistency
        annotate_audio(
            audio_path=audio_path,
            output_dir=output_dir,
            event_prompts=event_prompts,
            event_threshold=event_threshold,
            sound_prompts=sound_prompts,
            sound_threshold=sound_threshold,
            model_name=clap_model,
            device=device
        )
    
    # TODO: 5. Finalization - Update results_summary.json
    if verbose: typer.echo("📝 Updating results_summary.json (Not yet implemented)")

    if verbose:
        typer.echo(f"✅ Processing complete! Results saved to: {output_dir}")
    typer.secho("Processing complete!", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()