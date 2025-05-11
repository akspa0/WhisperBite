"""
AudioSegmenter core module.

This package contains the core components of the AudioSegmenter tool:
- audio_utils: Audio file processing with FFmpeg
- diarization: Speaker diarization using pyannote.audio
- clap_annotator: Sound event detection using CLAP models
- transcription: Speech transcription using Whisper models
"""

__version__ = "0.1.0" # Consider bumping version with new feature

# Expose key functions for easier access if desired, e.g.:
# from .audio_utils import prepare_audio
# from .diarization import initialize_diarization_pipeline, run_diarization_and_segmentation
# from .clap_annotator import initialize_clap_model, annotate_audio
# from .transcription import initialize_whisper_model, transcribe_segment, save_transcription_files 