# WhisperBite Modular CLAP Pipeline

WhisperBite is a modular, extensible audio processing framework for generating richly annotated, accessible, and searchable transcripts from audio or video. The 2025 refactor introduces a fully modular, rule-driven architecture with a focus on reproducibility, extensibility, and workflow sharing.

## Features
- **Modular Pipeline:** Each processing step (normalization, CLAP event detection, segmentation, diarization, transcription, soundbite extraction, output writing) is a separate module.
- **CLAP-Driven Segmentation:** Uses CLAP to detect events (e.g., speech, ringing, hang-up) and segment audio for downstream processing.
- **Speaker Diarization:** Accurate speaker separation using pyannote.audio.
- **Transcription:** High-quality transcription using OpenAI Whisper, with optional word-level timestamps.
- **Soundbite Extraction:** Extracts per-event and per-speaker soundbites with canonical naming.
- **Reproducible Outputs:** All runs produce a master transcript, per-segment TXT files, YAML metadata, and organized output directories.
- **Extensible:** Add new modules, rules, or workflows without breaking existing functionality.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/WhisperBite.git
   cd WhisperBite
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install system dependencies:**
   - [FFmpeg](https://ffmpeg.org/download.html) (required for audio processing)
   - [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (optional, for GPU acceleration)

4. **(Optional) Hugging Face Token:**
   - For speaker diarization, obtain a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Usage

### Command-Line Interface (CLI)

The modular CLAP pipeline can be run directly from the command line:

```bash
python app.py --input path/to/audio.wav --output_dir ./whisper_output --hf_token YOUR_HF_TOKEN --model base --enable_vocal_separation --enable_word_extraction
```

**Arguments:**
- `--input` (required): Path to the input audio or video file.
- `--output_dir` (required): Directory to save all outputs.
- `--hf_token`: Hugging Face token for pyannote diarization (required for speaker separation).
- `--model`: Whisper model name (default: `base`). Options: `tiny`, `base`, `small`, `medium`, `large`, etc.
- `--enable_vocal_separation`: Enable Demucs vocal separation (optional).
- `--enable_word_extraction`: Enable word-level timestamp extraction (optional).

**Example:**
```bash
python app.py --input test_audio/sample.wav --output_dir ./whisper_output --hf_token hf_xxx --model medium --enable_vocal_separation
```

### Output Structure

Each run creates a unique output directory with the following structure:

```
<output_dir>/
  master_transcript.txt
  results.yaml
  <Speaker_X>_transcriptions/
    0000_hello_this_is_bob.txt
    0000_hello_this_is_bob.wav
    ...
  <Speaker_X>_full_transcript.txt
  ...
```
- **master_transcript.txt:** Global transcript with all segments, speakers, and timings.
- **results.yaml:** YAML metadata summary of the run.
- **<Speaker_X>_transcriptions/**: Per-speaker segment audio and transcripts.
- **<Speaker_X>_full_transcript.txt:** Full transcript for each speaker.

### Modular Architecture

- All processing steps are implemented as independent modules in `modules/`.
- The pipeline is orchestrated by the `ClapPipelineManager` in `core/clap_pipeline_manager.py`.
- The module registry allows dynamic construction and execution of workflows.

### Extending the Pipeline
- Add new modules to `modules/` and register them in `app.py`.
- Define new workflows or rules in YAML or Python.
- See `config/presets.py` for workflow configuration examples.

## Development & Testing
- All modules are independently testable.
- To run the pipeline on a test file:
  ```bash
  python app.py --input test_audio/sample.wav --output_dir ./whisper_output --hf_token hf_xxx
  ```
- Outputs will be saved in the specified output directory.

## Requirements
- Python 3.8+
- PyTorch, pyannote.audio, openai-whisper, transformers, pydub, soundfile, ffmpeg, and other dependencies in `requirements.txt`.
- GPU recommended for best performance.

## License
MIT License

## Acknowledgments
- [OpenAI Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Demucs](https://github.com/facebookresearch/demucs)
- [CLAP](https://github.com/LAION-AI/CLAP)
