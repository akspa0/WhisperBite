import os
import ffmpeg
import numpy as np
import tempfile
import subprocess
import shutil
import platform
from pathlib import Path
from typing import Tuple

def check_ffmpeg_available():
    """Verify FFmpeg is available in the system."""
    try:
        # Check if ffmpeg command is available
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found in PATH")
            
        # Run version check
        subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        raise RuntimeError("FFmpeg not found or not working properly")

def convert_windows_path(path_str: str) -> str:
    """Convert Windows-style paths to WSL compatible paths if needed."""
    # Replace backslashes with forward slashes
    if '\\' in path_str:
        path_str = path_str.replace('\\', '/')
    
    # If running in WSL, handle Windows paths
    if platform.system() == 'Linux' and 'microsoft' in platform.release().lower():
        # Already processed by the CLI path normalizer
        pass
        
    return path_str

def ensure_valid_path(path_str: str) -> str:
    """Ensure the path is absolute and exists."""
    # Handle Windows-style paths in WSL
    path_str = convert_windows_path(path_str)
    
    # Normalize path
    path = Path(path_str).expanduser().resolve()
    
    # Check if input file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return str(path)

def load_audio(input_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file using ffmpeg, return (samples, sample_rate)"""
    try:
        # First, check if FFmpeg is available
        check_ffmpeg_available()
        
        # Handle Windows-style paths in WSL
        input_path = convert_windows_path(input_path)
        
        # Ensure input path is valid
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        
        # Get audio stream info
        probe = ffmpeg.probe(input_path)
        audio_stream = next(
            (s for s in probe['streams'] if s['codec_type'] == 'audio'), 
            None
        )
        if not audio_stream:
            raise ValueError("No audio stream found in file")
        
        sample_rate = int(audio_stream['sample_rate'])
        
        # Read audio into numpy array
        out, _ = (
            ffmpeg
            .input(input_path)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        return np.frombuffer(out, np.float32), sample_rate
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
        raise RuntimeError(f"FFmpeg error: {error_message}\nInput path: {input_path}")

def normalize_audio(input_path: str, output_path: str, target_lufs: float = -23.0):
    """EBU R128 loudness normalization using ffmpeg"""
    # Handle Windows-style paths in WSL
    input_path = convert_windows_path(input_path)
    
    # Ensure input path is valid
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    try:
        (
            ffmpeg
            .input(input_path)
            .filter('loudnorm', i=target_lufs)
            .output(output_path)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
        raise RuntimeError(f"FFmpeg normalization error: {error_message}\nInput: {input_path}, Output: {output_path}")

def resample_audio(input_path: str, output_path: str, target_sr: int = 16000):
    """High-quality resampling using ffmpeg"""
    # Handle Windows-style paths in WSL
    input_path = convert_windows_path(input_path)
    
    # Ensure input path is valid
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    try:
        (
            ffmpeg
            .input(input_path)
            .filter('aresample', target_sr)
            .output(output_path, ar=target_sr)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
        raise RuntimeError(f"FFmpeg resampling error: {error_message}\nInput: {input_path}, Output: {output_path}")

def prepare_audio(
    input_path: str, 
    output_dir: str, 
    normalize: bool = True,
    target_lufs: float = -23.0,
    force_resample: bool = True
) -> str:
    """Full preprocessing pipeline with FFmpeg"""
    # Verify FFmpeg is available
    check_ffmpeg_available()
    
    # Handle Windows-style paths in WSL
    input_path = convert_windows_path(input_path)
    
    # Print diagnostic info
    print(f"Processing file: {input_path}")
    print(f"Output directory: {output_dir}")
    
    # Verify input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prepared_audio.wav")
    
    try:
        # Use a persistent temp directory to avoid WSL file access issues
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a temporary file in our temp directory instead of /tmp
        temp_norm_path = os.path.join(temp_dir, "normalized_audio.wav")
        
        # Normalization stage
        if normalize:
            # Use a direct output file instead of a NamedTemporaryFile
            normalize_audio(input_path, temp_norm_path, target_lufs=target_lufs)
            input_path = temp_norm_path
        
        # Resampling stage
        if force_resample:
            resample_audio(input_path, output_path)
        else:
            ffmpeg.input(input_path).output(output_path).run(quiet=True)
        
        # Clean up temporary files if needed
        if os.path.exists(temp_norm_path):
            try:
                os.remove(temp_norm_path)
            except:
                pass  # Don't fail if temp file cleanup fails
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error preparing audio: {str(e)}") 