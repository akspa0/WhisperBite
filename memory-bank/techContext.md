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
*   **Output Format:** **Master transcript is `master_transcript.yaml`, containing metadata and structured segment data. Outputs are organized into a unique, timestamped directory per run, with step-specific subdirectories (e.g., `normalized/`, `demucs/`, `speakers/`, `events/`).**
*   Speaker Labels: Format is `S0`, `S1`, etc. **If `--split_stereo` is used, labels become `S0_L`, `S0_R`, etc.**
*   **Sound Detection:** Uses **CLAP model** from `transformers` to compare audio embeddings against text prompts for target sounds.
*   **Event Detection:** Also uses **CLAP model** for detecting broader event categories.
*   Some internal parameters in `whisperBite.py`

# Technical Context

## Core Technologies

### Machine Learning Models
1. **Whisper** (openai-whisper>=20231117)
   - Speech recognition and transcription
   - Multiple model sizes supported
   - Word-level timestamp capability

2. **Pyannote Audio** (pyannote.audio>=3.1.1)
   - Speaker diarization
   - Automatic speaker count detection
   - Pipeline-based processing

3. **CLAP** (via transformers>=4.35.2)
   - Sound event detection
   - Custom prompt support
   - High-confidence detection thresholds

### Audio Processing
1. **PyDub** (pydub>=0.25.1)
   - Audio file manipulation
   - Format conversion
   - Channel splitting

2. **FFmpeg** (ffmpeg-python>=0.2.0)
   - Audio extraction from video
   - Audio normalization
   - Format conversion

3. **Demucs** (demucs>=4.0.0)
   - Optional vocal separation
   - Enhanced sound detection

### Dependencies
```
gradio>=3.50.0          # Web UI
torch>=2.1.0           # Deep learning backend
torchaudio>=2.1.0      # Audio processing
librosa>=0.10.1        # Audio analysis
numpy>=1.24.3          # Numerical operations
soundfile>=0.12.1      # Audio file handling
scipy>=1.11.4          # Signal processing
PyYAML>=6.0.1         # Output formatting
yt-dlp>=2023.12.30    # URL downloading
```

## Technical Constraints

### Hardware Requirements
- GPU recommended for optimal performance
- Sufficient RAM for audio processing (8GB minimum)
- Adequate storage for temporary files and outputs

### Processing Limitations
- Maximum audio file size dependent on available memory
- Processing time scales with audio length and model size
- Second pass diarization increases processing time

### Input/Output
- Supported input formats: wav, mp3, m4a, flac, aac, mp4, mov, avi, mkv, webm
- Output format: WAV for audio segments
- Structured YAML for metadata and transcriptions

## Development Setup
1. Python 3.8+ required
2. CUDA toolkit for GPU support
3. FFmpeg installation required
4. Virtual environment recommended