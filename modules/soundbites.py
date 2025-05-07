import os
import re
import logging
from pydub import AudioSegment

def extract_soundbites(audio_obj, config=None):
    """
    Extracts soundbites from a segment based on event annotations in audio_obj.metadata['segment_annotations'].
    Saves soundbites to config['stage_dirs']['soundbites'] if present, else config['output_dir'], and returns updated audio_obj with provenance.
    """
    logger = logging.getLogger(__name__)
    segment_audio_path = audio_obj.path
    segment_annotations = audio_obj.metadata.get('segment_annotations', {})
    output_dir = config.get('stage_dirs', {}).get('soundbites', config.get('output_dir', '.'))
    base_filename = os.path.splitext(os.path.basename(segment_audio_path))[0]
    min_duration_s = config.get('min_duration_s', 0.2)
    padding_ms = config.get('padding_ms', 150)
    confidence_threshold = config.get('confidence_threshold', 0.3)
    os.makedirs(output_dir, exist_ok=True)
    soundbite_paths = []
    soundbite_index = 0
    logger.info(f"Starting soundbite extraction for {segment_audio_path}")
    if not segment_annotations:
        logger.warning(f"No segment_annotations found for {segment_audio_path}; skipping soundbite extraction.")
        audio_obj.add_provenance('extract_soundbites', {'soundbite_paths': []})
        return audio_obj
    try:
        segment_audio = AudioSegment.from_file(segment_audio_path)
        segment_duration_ms = len(segment_audio)
    except Exception as e:
        logger.error(f"Could not load segment audio {segment_audio_path} for soundbite extraction: {e}")
        return audio_obj
    for event_type, detections in segment_annotations.items():
        for detection in detections:
            try:
                confidence = detection.get('confidence', 0)
                if confidence < confidence_threshold:
                    continue
                start_s = detection['start']
                end_s = detection['end']
                duration_s = end_s - start_s
                if duration_s < min_duration_s:
                    continue
                extract_start_ms = max(0, int(start_s * 1000) - padding_ms)
                extract_end_ms = min(segment_duration_ms, int(end_s * 1000) + padding_ms)
                soundbite_audio = segment_audio[extract_start_ms:extract_end_ms]
                safe_event_type = re.sub(r'[\s/:]', '_', event_type)
                soundbite_filename = f"[{safe_event_type}]_{base_filename}_{int(start_s)}_{int(end_s)}_{soundbite_index:03d}.wav"
                soundbite_path = os.path.join(output_dir, soundbite_filename)
                fade_duration = min(50, soundbite_audio.duration_seconds * 1000 / 4)
                soundbite_audio = soundbite_audio.fade_in(int(fade_duration)).fade_out(int(fade_duration))
                soundbite_audio.export(soundbite_path, format="wav")
                soundbite_paths.append(soundbite_path)
                soundbite_index += 1
                logger.debug(f"  -> Extracted soundbite: {soundbite_path}")
            except Exception as e:
                logger.warning(f"Error extracting soundbite for detection {detection}: {e}")
                continue
    logger.info(f"Extracted {len(soundbite_paths)} soundbites for {os.path.basename(segment_audio_path)}.")
    audio_obj.add_provenance('extract_soundbites', {'soundbite_paths': soundbite_paths})
    return audio_obj 