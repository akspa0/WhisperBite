import logging
import os
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment

def format_speaker_label(label):
    if isinstance(label, int):
        return f"S{label}"
    elif isinstance(label, str):
        parts = label.split('_')
        num_part = parts[-1]
        try:
            num = int(num_part)
            return f"S{num}"
        except ValueError:
            return label
    else:
        return str(label)

def detect_optimal_speakers(diarization_pipeline, audio_file, min_speakers=1, max_speakers=10):
    best_score = -float('inf')
    best_num_speakers = 2
    for num_speakers in range(min_speakers, min(max_speakers + 1, 6)):
        try:
            diarization = diarization_pipeline(audio_file, num_speakers=num_speakers)
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({'start': turn.start, 'end': turn.end, 'duration': turn.end - turn.start, 'speaker': speaker})
            avg_segment_duration = sum(s['duration'] for s in segments) / len(segments) if segments else 0
            num_speaker_changes = len(segments) - 1
            score = avg_segment_duration * 10 - num_speaker_changes * 0.1
            if score > best_score:
                best_score = score
                best_num_speakers = num_speakers
        except Exception as e:
            continue
    return best_num_speakers

def diarize_speakers(audio_obj, config=None):
    """
    Perform speaker diarization using pyannote.audio Pipeline. Stores results in audio_obj.metadata['diarization'].
    """
    logger = logging.getLogger(__name__)
    audio_path = audio_obj.path
    hf_token = config.get('hf_token')
    min_speakers = config.get('min_speakers', 1)
    max_speakers = config.get('max_speakers', 6)
    try:
        logger.info(f"Starting diarization for {audio_path}")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline.to(device)
        actual_num_speakers = detect_optimal_speakers(pipeline, audio_path, min_speakers, max_speakers)
        diarization = pipeline(audio_path, num_speakers=actual_num_speakers)
        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if duration < 0.2:
                continue
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append({'start': turn.start, 'end': turn.end, 'duration': duration})
        segment_info = {}
        segment_counter = 0
        audio = AudioSegment.from_file(audio_path)
        output_dir = config.get('output_dir', '.')
        diarization_dir = os.path.join(output_dir, 'diarization')
        os.makedirs(diarization_dir, exist_ok=True)
        for speaker_raw, segments in speaker_segments.items():
            formatted_speaker = format_speaker_label(speaker_raw)
            speaker_dir = os.path.join(diarization_dir, formatted_speaker)
            os.makedirs(speaker_dir, exist_ok=True)
            if formatted_speaker not in segment_info:
                segment_info[formatted_speaker] = []
            segments.sort(key=lambda x: x['start'])
            merged_segments = []
            if segments:
                current = segments[0]
                for next_segment in segments[1:]:
                    gap = next_segment['start'] - current['end']
                    if gap < 0.5:
                        current['end'] = next_segment['end']
                        current['duration'] = current['end'] - current['start']
                    else:
                        merged_segments.append(current)
                        current = next_segment
                merged_segments.append(current)
            for i, segment in enumerate(merged_segments):
                segment_audio = audio[segment['start'] * 1000:segment['end'] * 1000]
                fade_duration = min(100, segment_audio.duration_seconds * 1000 / 4)
                segment_audio = segment_audio.fade_in(int(fade_duration)).fade_out(int(fade_duration))
                segment_filename = f"{segment_counter:04d}_segment_{int(segment['start'])}_{int(segment['end'])}_{int(segment['duration'])}.wav"
                segment_path = os.path.join(speaker_dir, segment_filename)
                segment_audio.export(segment_path, format="wav")
                segment_info[formatted_speaker].append({
                    'path': segment_path,
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['duration'],
                    'sequence': segment_counter
                })
                segment_counter += 1
        audio_obj.metadata['diarization'] = segment_info
        audio_obj.add_provenance('diarize_speakers', {'speakers': list(segment_info.keys()), 'diarization_dir': diarization_dir})
        logger.info(f"Diarization complete. Speakers: {list(segment_info.keys())}, segments: {segment_counter}")
    except Exception as e:
        logger.error(f"Diarization failed: {e}", exc_info=True)
        audio_obj.metadata['diarization'] = {}
    return audio_obj 