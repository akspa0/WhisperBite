import os
import logging
import subprocess
import tempfile
import shutil
import glob

def separate_vocals_with_demucs(input_audio, output_dir, model="htdemucs"):
    """
    Run Demucs vocal separation and place all output files directly in output_dir (no nested subfolders).
    If Demucs creates a nested subfolder, move the files up and remove the subfolder.
    
    Args:
        input_audio (str): Path to input audio file
        output_dir (str): Directory to save output
        model (str): Demucs model to use (htdemucs, htdemucs_ft, mdx, mdx_extra)
        
    Returns:
        tuple(str, str): Paths to the extracted (vocals_file, no_vocals_file).
                         no_vocals_file will be None if not found/created.
    """
    # Prepare Demucs command
    demucs_cmd = [
        "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "-o", output_dir,
        "--segment", "7",
        "--filename", "{track}_{stem}.{ext}",  # Flatten output naming
        input_audio
    ]
    # Run Demucs
    subprocess.run(demucs_cmd, check=True)
    # Demucs will create output_dir/model/ with files like inputname_vocals.wav
    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    model_dir = os.path.join(output_dir, model)
    if os.path.exists(model_dir):
        # Move all *_vocals.wav and *_no_vocals.wav up to output_dir
        for f in glob.glob(os.path.join(model_dir, f"{base_name}_*.wav")):
            shutil.move(f, output_dir)
        # Remove the now-empty model subfolder
        try:
            shutil.rmtree(model_dir)
        except Exception:
            pass
    vocals_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
    no_vocals_path = os.path.join(output_dir, f"{base_name}_no_vocals.wav")
    return (
        vocals_path if os.path.exists(vocals_path) else None,
        no_vocals_path if os.path.exists(no_vocals_path) else None
    )

def enhance_vocals(vocals_file, output_dir):
    """
    Enhance extracted vocals with noise reduction and EQ.
    Requires ffmpeg with the afftdn filter.
    
    Args:
        vocals_file (str): Path to vocal file
        output_dir (str): Directory to save output
        
    Returns:
        str: Path to enhanced vocal file
    """
    try:
        base_name = os.path.splitext(os.path.basename(vocals_file))[0]
        enhanced_file = os.path.join(output_dir, f"{base_name}_enhanced.wav")
        
        # Apply mild noise reduction and EQ to enhance vocals
        cmd = [
            "ffmpeg", "-y",
            "-i", vocals_file,
            "-af", "afftdn=nf=-20,equalizer=f=200:t=h:width=200:g=-3,equalizer=f=3000:t=h:width=1000:g=3,compand=0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2",
            "-ar", "44100",
            enhanced_file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logging.info(f"Enhanced vocals saved to {enhanced_file}")
        return enhanced_file
    except Exception as e:
        logging.warning(f"Error enhancing vocals: {e}. Using original vocals.")
        return vocals_file

def export_audio_segments(audio_segments, output_dir, base_name, sample_rate=48000):
    """Export audio segments to WAV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export vocals
    vocals_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
    audio_segments[0].export(vocals_path, format="wav", parameters=[
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le"
    ])
    
    # Export non-vocals (instrumental/background)
    no_vocals_path = os.path.join(output_dir, f"{base_name}_no_vocals.wav")
    audio_segments[1].export(no_vocals_path, format="wav", parameters=[
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le"
    ])
    
    return vocals_path, no_vocals_path
