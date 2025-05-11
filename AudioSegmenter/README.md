# AudioSegmenter

A tool to process audio files for speaker diarization, speech separation, and sound event annotation.

## Features

- Speaker diarization (who spoke when) using `pyannote/speech-separation-ami-1.0`
- Speaker separation to extract individual voices from mixed audio
- Speaker-separated audio segments based on diarization results
- Sound event annotation using CLAP models
- Support for various audio/video formats using FFmpeg

## Installation

### System Requirements

- Python 3.8+
- FFmpeg (system binary)
- CUDA-capable GPU (recommended)

### Setup

1. Install system dependencies:
   ```bash
   # For Debian/Ubuntu/WSL
   sudo apt update && sudo apt install ffmpeg -y
   
   # For macOS
   brew install ffmpeg
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. HuggingFace token:
   - Set the `HF_TOKEN` environment variable with your HuggingFace token
   - You must accept the conditions for both:
     - `pyannote/speech-separation-ami-1.0` on HuggingFace
     - `pyannote/separation-ami-1.0` on HuggingFace

## Usage

### Command Line Interface

Basic usage:
```bash
python audiosegmenter_cli.py input.mp3 output_dir/
```

With options:
```bash
python audiosegmenter_cli.py \
  input.mp3 \
  output_dir/ \
  --num-speakers 2 \
  --event-prompts "speech,music,silence" \
  --sound-prompts "dog barking,doorbell,applause" \
  --verbose
```

### Full Options

```
Options:
  --normalize / --no-normalize  Enable audio normalization [default: normalize]
  --target-lufs FLOAT           Target LUFS level for normalization [default: -23.0]
  --force-resample / --no-force-resample
                                Force resample to 16kHz mono [default: force-resample]
  --hf-token TEXT               Hugging Face token (default: HF_TOKEN env var)
  --num-speakers INTEGER        Exact speaker count
  --min-speakers INTEGER        Minimum speaker count [default: 1]
  --max-speakers INTEGER        Maximum speaker count [default: 5]
  --clap-model TEXT             CLAP model name [default: microsoft/clap-htsat-unfused]
  --device TEXT                 Inference device (cuda/cpu) [default: cuda]
  --event-prompts TEXT          Event prompts (comma-separated)
  --event-threshold FLOAT       Event detection threshold [default: 0.5]
  --sound-prompts TEXT          Sound prompts (comma-separated)
  --sound-threshold FLOAT       Sound detection threshold [default: 0.3]
  -v, --verbose                 Verbose output
  --help                        Show this message and exit.
```

## Output Files

The tool generates the following outputs in the specified output directory:

- `prepared_audio.wav`: Normalized, resampled audio file
- `diarization.rttm`: Speaker diarization results
- `SPEAKER_XX.wav`: Separated audio for each speaker
- `speaker_segments/speaker_segments_manifest.json`: Detailed segments information
- `clap_events.json`: General event annotations
- `clap_sounds.json`: Specific sound annotations
- `clap_annotations.json`: Combined annotations

## Citation

Based on pyannote's work:
```
@inproceedings{Kalda24,
  author={Joonas Kalda and Clément Pagés and Ricard Marxer and Tanel Alumäe and Hervé Bredin},
  title={{PixIT: Joint Training of Speaker Diarization and Speech Separation from Real-world Multi-speaker Recordings}},
  year=2024,
  booktitle={Proc. Odyssey 2024},
}
``` 