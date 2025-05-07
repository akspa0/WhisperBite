# Tech Context: WhisperBite (2025 Modular Refactor)

## Core Technologies

- **Python 3** (core language)
- **PyTorch, pyannote.audio, openai-whisper, transformers (CLAP), pydub, soundfile, ffmpeg**
- **YAML** for workflow/config definition and output
- **Module Registry:** All modules are Python classes/functions with explicit input/output/settings, registered for workflow use.
- **Rule Engine:** Explicit, schema-validated rules (YAML or Python) map audio annotations to workflow steps.

## Modularization Benefits

- Each module is small, focused, and independently testable.
- Changes in one module do not affect others, reducing regressions.
- Easier onboarding, documentation, and extension.

## Path Forward
- Scaffold and migrate to a modular, multi-file codebase.
- Complete and document all components for CLAP-driven segmentation.
- Prepare for LLM-driven rule engine as an optional enhancement.

## Development Setup

```mermaid
flowchart TD
    VENV[Python venv] --> Install[Install requirements.txt]
    Install --> FFmpeg[Install ffmpeg/demucs]
    FFmpeg --> Token[Add Hugging Face token (for diarization)]
    Token --> Run[Run CLI or UI]
```

## Output & Sharing
- All runs produce a `config.yaml` (sans secrets) and full metadata for reproducibility and sharing.

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