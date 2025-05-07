import logging
import os

def transcribe_with_whisper(audio_obj, config=None):
    """
    Transcribe audio using OpenAI Whisper (if available). Save transcript to metadata and provenance, and output a TXT file in the transcription subfolder if possible.
    If the audio is longer than max_chunk_seconds, split into chunks and concatenate results.
    """
    logger = logging.getLogger(__name__)
    try:
        import whisper
        from pydub import AudioSegment
    except ImportError:
        logger.warning("openai-whisper or pydub not installed; skipping transcription.")
        audio_obj.add_provenance('transcribe_with_whisper', {'warning': 'openai-whisper or pydub not installed'})
        return audio_obj
    model_name = config.get('model', 'base')
    transcription_dir = config.get('stage_dirs', {}).get('transcription', config.get('output_dir', '.'))
    os.makedirs(transcription_dir, exist_ok=True)
    audio_path = audio_obj.path
    model = whisper.load_model(model_name)
    max_chunk_seconds = config.get('max_chunk_seconds', 30)
    overlap_seconds = config.get('chunk_overlap_seconds', 2)
    audio = AudioSegment.from_file(audio_path)
    duration_s = len(audio) / 1000.0
    transcript = ''
    segments = []
    if duration_s > max_chunk_seconds:
        logger.info(f"Audio is {duration_s:.1f}s, chunking for Whisper transcription.")
        chunk_starts = list(range(0, int(duration_s), max_chunk_seconds - overlap_seconds))
        for i, start in enumerate(chunk_starts):
            end = min(start + max_chunk_seconds, duration_s)
            chunk = audio[start*1000:end*1000]
            chunk_path = os.path.join(transcription_dir, f"chunk_{i:03d}.wav")
            chunk.export(chunk_path, format="wav")
            result = model.transcribe(chunk_path)
            chunk_text = result.get('text', '')
            chunk_segments = result.get('segments', [])
            # Adjust segment times to global
            for seg in chunk_segments:
                seg['start'] += start
                seg['end'] += start
            transcript += chunk_text + '\n'
            segments.extend(chunk_segments)
        logger.info(f"Transcribed {len(chunk_starts)} chunks, total length {duration_s:.1f}s.")
    else:
        logger.info(f"Transcribing {audio_path} with Whisper model {model_name}")
        result = model.transcribe(audio_path)
        transcript = result.get('text', '')
        segments = result.get('segments', [])
    audio_obj.metadata['transcription'] = {'text': transcript, 'segments': segments}
    audio_obj.add_provenance('transcribe_with_whisper', {'model': model_name, 'output': transcript[:100]})
    # Save transcript to TXT file
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    transcript_path = os.path.join(transcription_dir, f"{base_name}_transcript.txt")
    try:
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        logger.info(f"Saved transcript to {transcript_path}")
    except Exception as e:
        logger.warning(f"Failed to save transcript: {e}")
    return audio_obj 