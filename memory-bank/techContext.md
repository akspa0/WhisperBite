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
*   **`tensorflow`**: Required for YAMNet sound detection.
*   **`tensorflow_hub`**: Used to load the YAMNet model.
*   **`librosa`**: Used for audio loading and resampling in sound detection.
*   **`soundfile`**: Often required as a backend by `librosa`.
*   Standard libraries: `logging`, `os`, `sys`, `datetime`, `json`, `subprocess`, `shutil`, `glob`, `tempfile`, `re`.

**External Dependencies (Command Line Tools):**
*   `ffmpeg`: **Required** for audio normalization and video audio extraction.
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
    *   `--auto_speakers`: Toggle automatic speaker count detection (default: off).
    *   `--enable_vocal_separation`: Toggle Demucs vocal separation (default: off).
*   **Speaker Labels:** Output format is `S0`, `S1`, etc.
*   Some internal parameters in `whisperBite.py` are hardcoded (LUFS target, segment merge gap, word padding, second pass thresholds).
*   Demucs parameters in `vocal_separation.py` are set (`--segment 7`, `--overlap 0.25`). **Segment size adjusted for htdemucs model limits.**
*   Sound detection parameters in `sound_detection.py` are set (YAMNet model handle, sample rate, detection threshold, target classes). 