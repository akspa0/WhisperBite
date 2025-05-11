# System Patterns

## System Architecture

- Modular pipeline architecture with distinct components:
  - Audio preprocessing (`core/audio_utils.py`)
  - Speaker diarization (`core/diarization.py`)
  - Audio segmentation (currently within `core/diarization.py`)
  - **NEW:** Speech transcription (`core/transcription.py`)
  - Sound event annotation (`core/annotator_clap.py`)
- CLI interface (`audiosegmenter_cli.py`) for configuration and execution.
- JSON-based manifests and structured file outputs for interoperability and downstream consumption.

## Key Technical Decisions

- Using `pyannote/speaker-diarization-3.1` for speaker diarization.
- **NEW:** Using `openai/whisper-large-v3` (via `transformers`) for speech transcription.
- CLAP models (e.g., `microsoft/clap-htsat-unfused` via `transformers`) for sound event annotation.
- Python as implementation language.
- Leveraging `ffmpeg` for core audio processing tasks (loading, normalization, resampling).

## Design Patterns in Use

- **Pipeline Pattern:** For sequential processing stages (preprocess -> diarize -> segment -> transcribe -> annotate).
- **Strategy Pattern:** Could be considered for selecting different models (e.g., different Whisper model sizes, different diarization or CLAP models if supported in the future) via configuration.
- **Factory Pattern:** Implicitly used for model initialization within each respective module (e.g., `initialize_diarization_pipeline`, `initialize_clap_model`, and upcoming `initialize_whisper_model`). 