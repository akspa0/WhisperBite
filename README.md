# AudioSegmenter: Advanced Audio Analysis Tool

AudioSegmenter is a Python CLI tool designed to perform comprehensive analysis on audio files. It leverages state-of-the-art machine learning models to provide speaker diarization, audio segmentation, speech transcription, and sound event detection.

## Features

*   **Speaker Diarization:** Identifies who spoke when in an audio file (`pyannote/speaker-diarization-3.1`).
*   **Audio Segmentation:** Splits the audio into separate WAV files based on speaker turns.
*   **Speech Transcription:** Transcribes each speaker segment using Whisper (`openai/whisper-large-v3` by default) and reconstructs a full, speaker-attributed transcript with word-level timestamps.
*   **Sound Event Annotation:** Detects specified sound events (e.g., music, noise) and specific sounds (e.g., dog barking, doorbell) using CLAP models (`microsoft/clap-htsat-unfused` by default).
*   **Audio Preprocessing:** Includes options for loudness normalization (e.g., to YouTube standards -14 LUFS) and resampling to 16kHz mono for model compatibility.

## Core Technologies

*   **Diarization:** `pyannote.audio` (Hugging Face `pyannote/speaker-diarization-3.1`)
*   **Transcription:** `openai-whisper` (via Hugging Face `transformers`, `openai/whisper-large-v3` model)
*   **Sound Annotation:** CLAP (Contrastive Language-Audio Pretraining) models (via Hugging Face `transformers`, `microsoft/clap-htsat-unfused` model)
*   **Audio Processing:** `ffmpeg` (via `ffmpeg-python`)
*   **CLI:** `Typer`

## Prerequisites

*   Python 3.8+ (Python 3.9+ recommended)
*   **FFmpeg:** Must be installed on your system and accessible via the system PATH.
    *   Linux/WSL: `sudo apt update && sudo apt install ffmpeg`
    *   macOS: `brew install ffmpeg`
    *   Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Setup

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone <repository-url>
    # cd AudioSegmenter
    ```

2.  **Create and Activate a Virtual Environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Hugging Face User Access Token (Required for Pyannote):**
    *   The `pyannote/speaker-diarization-3.1` model requires authentication with a Hugging Face token and agreement to its user conditions on the Hugging Face Hub.
    *   Obtain a token from [Hugging Face settings](https://huggingface.co/settings/tokens).
    *   You can provide the token via the `--hf-token YOUR_TOKEN` CLI argument or by setting the `HF_TOKEN` environment variable (e.g., in a `.env` file at the root of this tool, which `python-dotenv` will load).
        ```
        # Example .env file content:
        HF_TOKEN=your_hugging_face_token_here
        ```

## Usage

The main command is `process`. Use `python audiosegmenter_cli.py process --help` for all options.

```bash
python audiosegmenter_cli.py process [OPTIONS] INPUT_PATH OUTPUT_DIR
```

**Key Arguments & Options:**

*   `INPUT_PATH`: Path to the input audio or video file (required).
*   `OUTPUT_DIR`: Directory to save all processing results (required).

*   **Audio Preprocessing:**
    *   `--normalize` / `--no-normalize`: Enable/disable loudness normalization (default: enabled).
    *   `--target-lufs FLOAT`: Target LUFS for normalization (default: -14.0).
    *   `--resample` / `--no-resample`: Enable/disable resampling to 16kHz mono (default: enabled).

*   **Diarization:**
    *   `--hf-token TEXT`: Your Hugging Face token.
    *   `--num-speakers INTEGER`: Exact number of speakers.
    *   `--min-speakers INTEGER`: Minimum number of speakers (default: 1).
    *   `--max-speakers INTEGER`: Maximum number of speakers (default: 5).

*   **Transcription:**
    *   `--transcribe` / `--no-transcribe`: Enable/disable transcription (default: enabled).
    *   `--whisper-model-name TEXT`: Whisper model to use (default: `openai/whisper-large-v3`).
    *   `--transcription-language TEXT`: Language code for Whisper (e.g., `en`, `fr`; default: auto-detect).
    *   `--word-timestamps` / `--no-word-timestamps`: Enable/disable word-level timestamps (default: enabled).

*   **CLAP Sound Annotation:**
    *   `--clap-model TEXT`: CLAP model to use (default: `microsoft/clap-htsat-unfused`).
    *   `--event-prompts TEXT`: Comma-separated general event prompts (e.g., "speech,music,noise").
    *   `--event-threshold FLOAT`: Detection threshold for general events (default: 0.5).
    *   `--sound-prompts TEXT`: Comma-separated specific sound prompts (e.g., "dog barking,doorbell").
    *   `--sound-threshold FLOAT`: Detection threshold for specific sounds (default: 0.3).

*   **General:**
    *   `--device TEXT`: Inference device (`cuda`, `cpu`, `mps`; default: `cuda`, falls back to `cpu` if unavailable).
    *   `--verbose` / `-v`: Enable verbose output.

### Example Command:

```bash
python audiosegmenter_cli.py process "path/to/your/audio.mp3" "./output_results" \
    --hf-token "YOUR_HF_TOKEN_HERE" \
    --transcribe \
    --event-prompts "speech,music,noise,applause" \
    --verbose
```

## Output Structure

All outputs are saved within the specified `OUTPUT_DIR`:

*   `prepared_audio.wav`: The preprocessed audio file (normalized, 16kHz mono) used for analysis.
*   `diarization.rttm`: Standard RTTM file detailing speaker turns.
*   `speaker_segments/`: Contains subdirectories for each identified speaker (`SPEAKER_00`, `SPEAKER_01`, etc.), with their segmented audio turns as WAV files (e.g., `SPEAKER_00_turn_0.wav`).
*   `speaker_segments/speaker_segments_manifest.json`: JSON file listing all speaker segments, their speaker ID, start/end times, duration, and relative file paths.
*   `transcripts/` (if transcription enabled):
    *   `speaker_segments/`: Mirrors the structure of `speaker_segments/` but contains `.txt` (plain text) and `.json` (detailed with word timestamps) transcription files for each audio segment.
    *   `full_transcript.json`: A JSON array of all transcribed utterances, sorted by time, with speaker ID, global start/end times, text, and word timestamps.
    *   `full_transcript.txt`: A human-readable version of the full transcript.
*   `clap_events.json`: (If event prompts provided) Detected general sound events with timestamps and confidence.
*   `clap_sounds.json`: (If sound prompts provided) Detected specific sounds with timestamps and confidence.
*   `processing_log.txt`: (Planned - detailed log of the tool's execution).
*   `results_summary.json`: (Planned - summary of all parameters and output file paths).

## Key Considerations

*   **Model Downloads:** On the first run, the necessary Hugging Face models (Pyannote, Whisper, CLAP) will be downloaded. This can take time and requires internet access. Models are cached locally for subsequent runs.
*   **Resource Usage:** 
    *   **Whisper `large-v3`** is a large model and requires significant RAM/VRAM and processing power for timely results. GPU usage (`--device cuda`) is highly recommended.
    *   Consider using smaller Whisper variants via `--whisper-model-name` (e.g., `openai/whisper-base`) if resources are limited, at the cost of transcription accuracy.
    *   The `accelerate` library (included in `requirements.txt`) helps with loading large models.
*   **Hugging Face Token:** Essential for `pyannote/speaker-diarization-3.1` due to its gated access.

## License

(Specify your project's license here, e.g., MIT License) 