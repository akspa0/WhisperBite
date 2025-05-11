# audiosegmenter_cli.py
import typer
from typing import List, Optional
import os
import sys
import subprocess
import ffmpeg
import re
from pathlib import Path, PureWindowsPath
from core.audio_utils import prepare_audio
from core.diarization import initialize_diarization_pipeline, run_diarization_and_segmentation
from core.clap_annotator import annotate_audio

app = typer.Typer(help="AudioSegmenter: Speaker diarization, separation, and sound annotation tool")

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
    normalize: bool = typer.Option(True, help="Enable audio normalization"),
    target_lufs: float = typer.Option(-14.0, help="Target LUFS level for normalization (e.g., -14.0 for YouTube)"),
    force_resample: bool = typer.Option(True, help="Force resample to 16kHz mono"),
    # Diarization and Separation
    hf_token: str = typer.Option(None, help="Hugging Face token (default: HF_TOKEN env var)"),
    num_speakers: Optional[int] = typer.Option(None, help="Exact speaker count"),
    min_speakers: int = typer.Option(1, help="Minimum speaker count"),
    max_speakers: int = typer.Option(5, help="Maximum speaker count"),
    # CLAP annotation
    clap_model: str = typer.Option("microsoft/clap-htsat-unfused", help="CLAP model name"),
    device: str = typer.Option("cuda", help="Inference device (cuda/cpu)"),
    event_prompts: Optional[List[str]] = typer.Option(None, help="Event prompts (comma-separated)"),
    event_threshold: float = typer.Option(0.5, help="Event detection threshold"),
    sound_prompts: Optional[List[str]] = typer.Option(None, help="Sound prompts (comma-separated)"), 
    sound_threshold: float = typer.Option(0.3, help="Sound detection threshold"),
    # Debug
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Process audio file through speaker diarization, separation, and sound event annotation."""
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
    
    # 2. Speaker diarization and separation
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
    
    # 3. Sound annotation (if prompts provided)
    if event_prompts or sound_prompts:
        if verbose:
            typer.echo("🎵 Running sound annotation...")
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
    
    if verbose:
        typer.echo(f"✅ Results saved to: {output_dir}")
    typer.secho("Processing complete!", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()