import os
import subprocess

def normalize_audio(audio_obj, config=None):
    """
    Normalize audio to -16 LUFS and resample to 48kHz WAV using ffmpeg.
    Updates audio_obj.path to the normalized, resampled file.
    Output is written to config['stage_dirs']['normalized'] if present, else config['output_dir'].
    """
    input_path = audio_obj.path
    output_dir = config.get('stage_dirs', {}).get('normalized', config.get('output_dir', '.'))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_normalized_48k.wav")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "48000", "-acodec", "pcm_s16le", output_file
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        audio_obj.path = output_file
        audio_obj.add_provenance('normalize_audio', {'output': output_file})
    except subprocess.CalledProcessError as e:
        audio_obj.add_provenance('normalize_audio', {'error': str(e)})
        raise RuntimeError(f"Error normalizing audio: {e}")
    return audio_obj 