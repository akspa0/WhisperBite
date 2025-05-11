# Technical Context

## Technologies Used

- **Core Libraries**:
  - `pyannote.audio` (version 3.1 or higher) for speaker diarization.
  - `transformers` (Hugging Face) for CLAP models and **NEW: Whisper speech-to-text models.**
  - `pydub` or `soundfile` for audio manipulation (though `ffmpeg-python` and `scipy` are also used for some aspects).
  - `ffmpeg-python` for direct `ffmpeg` bindings used in `audio_utils.py`.
  - `typer` for the CLI interface.
- **Models**:
  - Diarization: `pyannote/speaker-diarization-3.1`.
  - **NEW: Transcription: `openai/whisper-large-v3` (default).**
  - Annotation: CLAP models (e.g., `microsoft/clap-htsat-unfused` default).

## Development Setup

- Python environment (3.8+ recommended).
- Requires Hugging Face authentication token (`HF_TOKEN`) for Pyannote models and potentially for gated Hugging Face models.
- GPU (NVIDIA CUDA enabled) strongly recommended for model inference (Pyannote, CLAP, and especially Whisper `large-v3`).
- `ffmpeg` must be installed and available in the system PATH.

## Dependencies

- Core ML/Audio:
  - `torch`
  - `torchaudio`
  - `pyannote.audio`
  - `transformers` (ensure it includes sentencepiece for Whisper, e.g., `transformers[torch,sentencepiece]` or similar, `accelerate` is recommended for faster model loading, especially for large models like Whisper).
  - `ffmpeg-python`
- Audio Processing Helpers:
  - `pydub` (currently imported in `diarization.py` and `clap_annotator.py`)
  - `soundfile` (alternative, not explicitly used yet but good for WAV I/O)
  - `librosa` (used in `clap_annotator.py` for resampling)
  - `scipy` (used for `wavfile.write` in `diarization.py`)
- CLI:
  - `typer`

## Key Considerations for Transcription Module

- **Whisper Model Loading:** `openai/whisper-large-v3` can be slow to load and requires significant disk space and VRAM. The `accelerate` library can help with faster loading and better memory management across multiple GPUs if available.
- **Batch Processing:** For transcribing many small segments, batching (if supported by the `transformers` Whisper pipeline for inference on multiple files) could be more efficient than loading and transcribing one by one. This might be an optimization for later.
- **Word Timestamp Accuracy:** While generally good, the exact precision of word-level timestamps from Whisper can vary. For critical applications, this might need evaluation. 