import os
import logging
from core import AudioObject

def separate_vocals_with_demucs(audio_obj, config=None):
    """
    Run Demucs separation on the input audio, save both vocal and no_vocal tracks, and return a dict:
    {'vocal': vocal_audio_obj, 'no_vocal': no_vocal_audio_obj}
    Each is an AudioObject with correct path, metadata, and provenance.
    Output is written to config['stage_dirs']['demucs'] if present, else config['output_dir'].
    """
    logger = logging.getLogger(__name__)
    input_path = audio_obj.path
    output_dir = config.get('stage_dirs', {}).get('demucs', config.get('output_dir', '.'))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    vocal_path = os.path.join(output_dir, f"{base_name}_vocal.wav")
    no_vocal_path = os.path.join(output_dir, f"{base_name}_no_vocal.wav")
    # Placeholder: Replace with actual Demucs call
    # For now, just copy the input file to both outputs for structure
    import shutil
    try:
        shutil.copy(input_path, vocal_path)
        shutil.copy(input_path, no_vocal_path)
        logger.info(f"Demucs (placeholder): copied input to {vocal_path} and {no_vocal_path}")
    except Exception as e:
        logger.error(f"Demucs failed to copy files: {e}")
        raise
    vocal_audio_obj = AudioObject(vocal_path, metadata={'source': input_path, 'demucs_type': 'vocal'})
    no_vocal_audio_obj = AudioObject(no_vocal_path, metadata={'source': input_path, 'demucs_type': 'no_vocal'})
    vocal_audio_obj.add_provenance('separate_vocals_with_demucs', {'output': vocal_path})
    no_vocal_audio_obj.add_provenance('separate_vocals_with_demucs', {'output': no_vocal_path})
    return {'vocal': vocal_audio_obj, 'no_vocal': no_vocal_audio_obj} 