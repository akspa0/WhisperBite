CANONICAL_WORKFLOW = {
    'steps': [
        'normalize_audio',
        'annotate_with_clap',
        'segment_audio',
        'separate_vocals_with_demucs',
        'diarize_speakers',
        'transcribe_with_whisper',
        'extract_soundbites',
        'write_outputs',
    ]
} 