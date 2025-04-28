# Tech Context

**Core Language:** Python 3

**Key Libraries:**
*   **`gradio`**: Used by `app.py` to create the web UI.
*   `openai-whisper`: Core speech-to-text transcription (`whisperBite.py`).
*   `pyannote.audio`: Speaker diarization (`whisperBite.py`). Requires Hugging Face token.
*   `torch`: Required backend for Whisper and Pyannote (supports CPU/GPU).
*   `pydub`: Audio manipulation (`whisperBite.py`).
*   `argparse`: Command-line argument parsing.
*   `requests`: For downloading URLs (via `utils.py`).
*   `yt-dlp`: Used by `utils.download_audio` for URL downloads.
*   `demucs`: Optional library for vocal separation (installed via pip).
*   **`transformers`**: Used for CLAP-based sound event detection.
*   `librosa`: Used for audio loading and resampling (dependency for Whisper/sound detection).
*   `soundfile`: Often required as a backend by `librosa`.
*   **`PyYAML`**: For writing structured YAML output.
*   Standard libraries: `logging`, `os`, `sys`, `datetime`, `json`, `subprocess`, `shutil`, `glob`, `tempfile`, `re`, `yaml`.

**External Dependencies (Command Line Tools):**
*   `ffmpeg`/`ffprobe`: **Required** for audio normalization, video extraction, **and media info extraction.**
*   `demucs` (command-line): **Implicitly required** if `demucs` Python package is used for vocal separation. **Uses `--segment` and `--overlap` internally for large file processing.**

**Development Setup:**
*   Python environment with libraries installed from `requirements.txt`.
*   `ffmpeg` and optionally `demucs` installed and in PATH.
*   Internet access for model/audio downloads.
*   Hugging Face account and API token required for diarization.

**Execution:**
*   **Web UI:** `python app.py` (Optional args: `--public`, `--port`).
*   **CLI:** `python whisperBite.py`.

**Hardware:**
*   CUDA-compatible GPU recommended, CPU supported.

**Configuration:**
*   **Web UI (`app.py`):** Interactive Gradio components.
*   **CLI (`whisperBite.py`):** `argparse` arguments, including:
    *   `--enable_word_extraction`: Toggle word audio snippet generation (default: off).
    *   `--enable_second_pass`: Toggle diarization refinement pass (default: off).
    *   `--second_pass_min_duration`: Minimum segment length (seconds) for second pass (default: 5.0).
    *   `--auto_speakers`: Toggle automatic speaker count detection (default: off).
    *   `--enable_vocal_separation`: Toggle Demucs vocal separation (default: off).
    *   **`--split_stereo`**: Process L/R channels separately if input is stereo (default: off).
    *   **`--attempt_sound_detection`**: Toggle CLAP-based sound detection (default: off).
*   **Output Format:** **Master transcript is now `master_transcript.yaml`, containing metadata and structured segment data (potentially nested or split by channel).**
*   Speaker Labels: Format is `S0`, `S1`, etc. **If `--split_stereo` is used, labels become `S0_L`, `S0_R`, etc.**
*   **Sound Detection:** Uses **CLAP model** from `transformers` to compare audio embeddings against text prompts for target sounds (e.g., "telephone ringing").
*   Some internal parameters in `whisperBite.py`