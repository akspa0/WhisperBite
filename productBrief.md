# Specification: AudioSegmenter Tool

## 1. Purpose

To process an input audio file and produce:
*   Speaker diarization results (who spoke when) using `pyannote/speaker-diarization-3.1`.
*   Speaker-separated audio segments based on the diarization.
*   Timestamps and labels for detected sound events/annotations using a CLAP model.

This tool is designed to be a focused, modular component whose output can be consumed by other audio processing tools or pipelines.

## 2. Core Technologies

*   **Speaker Diarization:**
    *   **Library:** `pyannote.audio` (version 3.1 or higher)
        *   Homepage: [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
    *   **Pipeline:** `pyannote/speaker-diarization-3.1`
        *   Hugging Face Model Card: [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
        *   This pipeline ingests mono audio at 16kHz and outputs speaker diarization as an `Annotation` instance. It handles downmixing and resampling automatically.
*   **Sound/Event Annotation:**
    *   **Model Type:** CLAP (Contrastive Language-Audio Pretraining)
    *   **Library:** `transformers` (for loading CLAP models from Hugging Face) or a dedicated CLAP implementation.
    *   **Example CLAP Model:** [microsoft/clap-htsat-unfused](https://huggingface.co/microsoft/clap-htsat-unfused) (or other suitable CLAP models available on Hugging Face).

## 3. Inputs

*   **Primary Input:**
    *   Single audio file (common formats like WAV, MP3, FLAC, M4A).
    *   Video files (common formats like MP4, MOV, AVI, MKV) - audio will be extracted.
*   **Configuration Parameters** (e.g., via CLI arguments or a configuration file like YAML/JSON):
    *   `input_file_path`: (Required) Path to the audio/video file.
    *   `output_dir`: (Required) Directory to save all processing results.
    *   `hf_token`: (Required) Hugging Face User Access Token (for `pyannote/speaker-diarization-3.1` and potentially other gated models). Can be sourced from an environment variable.
    *   **Diarization Settings (for `pyannote/speaker-diarization-3.1`):**
        *   `num_speakers`: (Optional) Integer. Known number of speakers.
        *   `min_speakers`: (Optional) Integer. Minimum number of speakers if `num_speakers` is not set.
        *   `max_speakers`: (Optional) Integer. Maximum number of speakers if `num_speakers` is not set.
    *   **CLAP Annotation Settings:**
        *   `clap_model_name`: (Optional) String. Hugging Face identifier for the CLAP model (e.g., "microsoft/clap-htsat-unfused"). Defaults to a reasonable choice.
        *   `clap_event_prompts`: (Optional) List of strings. Text prompts for general event detection (e.g., \["speech", "music", "silence", "noise"]). If not provided, a default set may be used or this step skipped.
        *   `clap_event_threshold`: (Optional) Float. Detection threshold for general events (e.g., 0.5).
        *   `clap_sound_prompts`: (Optional) List of strings. Text prompts for specific sound detection (e.g., \["dog barking", "doorbell", "applause"]). If not provided, a default set may be used or this step skipped.
        *   `clap_sound_threshold`: (Optional) Float. Detection threshold for specific sounds (e.g., 0.3).
    *   `device`: (Optional) String. Device for model inference ("cuda" or "cpu"). Defaults to "cuda" if available, else "cpu".
    *   `normalize_audio`: (Optional) Boolean. Whether to normalize input audio loudness before processing. Defaults to `True`.
    *   `force_resample_to_16k`: (Optional) Boolean. While `pyannote/speaker-diarization-3.1` handles resampling, this flag could ensure an explicit resampling step to 16kHz mono before any processing, if desired for consistency with other steps like CLAP. Defaults to `True`.


## 4. Processing Pipeline

1.  **Setup & Initialization:**
    *   Load configurations.
    *   Initialize Pyannote diarization pipeline (`Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)`). Send to `device`.
    *   Initialize CLAP model and processor (e.g., `ClapModel.from_pretrained(clap_model_name)`, `ClapProcessor.from_pretrained(clap_model_name)`). Send to `device`.
2.  **Input Preprocessing (`core/audio_utils.py`):**
    *   Create unique `output_dir` if it doesn\'t exist.
    *   If input is video, extract audio to a temporary WAV file.
    *   Convert other input audio formats to a temporary WAV file if not already WAV.
    *   If `normalize_audio` is `True`, normalize the WAV file (e.g., to a standard LUFS level like -23 LUFS).
    *   Ensure the audio is 16kHz mono for Pyannote (though the pipeline handles this, an explicit step can be added if `force_resample_to_16k` is true). Let this be the `prepared_audio.wav`.
3.  **Speaker Diarization (`core/diarization.py`):**
    *   Run the Pyannote pipeline on `prepared_audio.wav` (or its in-memory representation: `{"waveform": waveform, "sample_rate": sample_rate}`):
        ```python
        # from pyannote.audio import Pipeline
        # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        # pipeline.to(torch.device(DEVICE))
        # diarization = pipeline(prepared_audio_path, num_speakers=NUM_SPEAKERS, min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
        ```
    *   The `diarization` object is a `pyannote.core.Annotation` instance.
    *   Save the RTTM output: `with open(output_rttm_path, "w") as rttm_file: diarization.write_rttm(rttm_file)`
4.  **Audio Segmentation by Speaker (`core/audio_utils.py`):**
    *   Iterate through the speaker turns in the `diarization` object (`for turn, _, speaker_label in diarization.itertracks(yield_label=True):`).
    *   For each turn (`turn.start`, `turn.end`), extract the corresponding audio segment from `prepared_audio.wav` (e.g., using `pydub` or `soundfile`).
    *   Save each segment to `output_dir/speaker_segments/<speaker_label>/<speaker_label>_turn_<N>.wav`.
    *   Create `speaker_segments.json` listing all segments, their speaker, start/end times, and relative file paths.
5.  **CLAP Event/Sound Annotation (`core/clap_annotator.py`):**
    *   **General Events:** If `clap_event_prompts` are provided:
        *   Process `prepared_audio.wav` with the CLAP model and `clap_event_prompts`.
        *   Filter detections by `clap_event_threshold`.
        *   Store results (start, end, label, confidence) in `clap_events.json`.
    *   **Specific Sounds:** If `clap_sound_prompts` are provided:
        *   Process `prepared_audio.wav` (or consider processing only non-speech regions if feasible for efficiency, though this adds complexity) with the CLAP model and `clap_sound_prompts`.
        *   Filter detections by `clap_sound_threshold`.
        *   Store results (start, end, label, confidence) in `clap_sounds.json`.
6.  **Finalization:**
    *   Create the main `manifest.json` (or `results.json`) file summarizing all parameters and output file paths.
    *   Write processing logs to `processing_log.txt`.

## 5. Outputs (Structure within `output_dir`)

*   `input_audio_info.json`: (Optional) Information about the original input audio (format, duration, channels, etc.).
*   `prepared_audio.wav`: (Optional, for inspection) The final audio file (e.g., normalized, 16kHz mono) used for processing.
*   `diarization.rttm`: Standard RTTM file from Pyannote, detailing speaker turns.
*   `speaker_segments/`:
    *   `SPEAKER_00/`:
        *   `SPEAKER_00_turn_0.wav`
        *   `SPEAKER_00_turn_1.wav`
        *   ...
    *   `SPEAKER_01/`:
        *   ...
    *   `speaker_segments_manifest.json`:
        ```json
        [
            {"speaker_id": "SPEAKER_00", "start_time": 0.5, "end_time": 10.2, "duration": 9.7, "file_path": "speaker_segments/SPEAKER_00/SPEAKER_00_turn_0.wav"},
            // ... more segments
        ]
        ```
*   `clap_events.json`: (If event prompts provided)
    ```json
    [
        {"start_time": 10.5, "end_time": 15.2, "label": "speech", "prompt_used": "speech", "confidence": 0.92},
        {"start_time": 15.8, "end_time": 20.0, "label": "music", "prompt_used": "music", "confidence": 0.75}
    ]
    ```
*   `clap_sounds.json`: (If sound prompts provided) Structure similar to `clap_events.json`.
*   `processing_log.txt`: Detailed log of the tool\'s execution.
*   `results_summary.json`:
    ```json
    {
        "processing_tool": "AudioSegmenter",
        "version": "0.1.0", // Tool version
        "input_file_path_original": "path/to/original_audio.mp4",
        "processing_parameters": {
            "hf_token_provided": true, // or false
            "num_speakers": null, // or integer value
            "min_speakers": 1,
            "max_speakers": 5,
            "clap_model_name": "microsoft/clap-htsat-unfused",
            "clap_event_prompts": ["speech", "music"],
            // ... other params
        },
        "diarization_stats": {
            "identified_speakers": ["SPEAKER_00", "SPEAKER_01"],
            "total_speech_duration": 120.5 // seconds
        },
        "outputs": {
            "rttm_file": "diarization.rttm",
            "speaker_segments_manifest": "speaker_segments/speaker_segments_manifest.json",
            "clap_events_file": "clap_events.json", // or null
            "clap_sounds_file": "clap_sounds.json", // or null
            "log_file": "processing_log.txt"
        },
        "processing_start_time": "YYYY-MM-DDTHH:MM:SSZ",
        "processing_end_time": "YYYY-MM-DDTHH:MM:SSZ",
        "processing_duration_seconds": 15.7,
        "status": "success", // or "failure"
        "error_message": null // or string message
    }
    ```

## 6. Proposed Python CLI Tool Structure

```
AudioSegmenter/
├── audiosegmenter_cli.py    # Main CLI script (using argparse or typer)
├── core/
│   ├── __init__.py
│   ├── audio_utils.py       # Audio loading, extraction, normalization, resampling, segmentation
│   ├── diarization_pyannote.py # Pyannote 3.1 diarization logic
│   ├── annotator_clap.py    # CLAP event/sound annotation logic
│   └── config_handler.py    # (Optional) For loading/validating config files
├── README.md                # Setup, usage, examples
├── requirements.txt         # e.g., pyannote.audio, torch, torchaudio, transformers, pydub (or soundfile)
└── .gitignore
```

## 7. Key Considerations

*   **User Authentication for Hugging Face:** Clearly document the need for `hf_token` and how to obtain/provide it (e.g., environment variable `HF_TOKEN` or direct argument). `pyannote/speaker-diarization-3.1` requires agreeing to user conditions on Hugging Face.
*   **Model Downloads:** Pyannote and CLAP models will be downloaded on first use. Ensure appropriate caching.
*   **Error Handling:** Implement try-except blocks for pipeline steps, model loading, and file operations. Log errors clearly.
*   **Dependencies:** Manage Python dependencies strictly in `requirements.txt`.
*   **License:** Consider the licenses of all used components (Pyannote.audio: MIT, specific models may vary but `pyannote/speaker-diarization-3.1` is MIT).

