# System Patterns

**Overall Architecture:**
*   Core processing pipeline (`whisperBite.py`).
*   Gradio web UI (`app.py`).
*   Conditional Two-Pass Pipeline.

**Key Components/Functions:**
*   `app.py`:
    *   `build_interface()`: Defines UI, accepting audio/video file types.
    *   `run_pipeline()`: Handles UI logic, calls `whisperBite.process_audio`.
*   `whisperBite.py`:
    *   `process_audio()`: Main orchestrator.
        *   Accepts `--split_stereo` flag.
        *   Determines actual input file (handles directory/URL).
        *   Calls `utils.get_media_info`.
        *   Checks if input is video, calls `extract_audio_from_video`.
        *   Calls normalization, optional vocal separation.
        *   **Checks audio channels (`utils.get_audio_channels`) and `split_stereo` flag.**
        *   **If `split_stereo` and 2 channels:**
            *   **Splits audio into L/R mono files.**
            *   **Runs diarization, slicing (with `_L`/`_R` suffix), and transcription independently on L and R channels.**
        *   **Else (mono or `split_stereo` disabled):**
            *   **Runs diarization, slicing, and transcription on the single input track.**
            *   **Conditionally calls `run_second_pass_diarization` (using `second_pass_min_duration`).**
        *   **Conditionally calls `sound_detection.detect_sound_events` on `no_vocals` track.**
        *   **Builds nested Python dictionary containing metadata (including media info, options, URL) and segments (either separate L/R lists or a single list potentially with nested refined segments).**
        *   **Writes the dictionary to `master_transcript.yaml` using `yaml.dump()`.**
        *   Calls `zip_results`.
        *   Cleans up temporary files and file logger.
    *   `extract_audio_from_video()`: Uses `ffmpeg`.
    *   `transcribe_with_whisper()`: Handles transcription, word extraction. **Accepts `speaker_suffix`.**
    *   `run_second_pass_diarization()`: Implements refinement pass. **Accepts `segment_min_duration`.**
    *   `format_speaker_label()`: Converts raw speaker labels.
    *   `slice_audio_by_speaker()`: Slices audio. **Accepts `speaker_suffix`.**
    *   Other helpers.
*   `vocal_separation.py`:
    *   `separate_vocals_with_demucs()`: Runs Demucs with chunking.
*   `sound_detection.py`:
    *   `detect_sound_events()`: **Loads CLAP model/processor. Processes audio in chunks, gets audio embeddings, compares to text prompt embeddings for target sounds, and returns detected event times/labels.**
*   `utils.py`: 
    *   Helpers (`sanitize_filename`, `download_audio`, `zip_results`).
    *   **`get_media_info()`: Uses `ffprobe` to get media metadata.**
    *   **`get_audio_channels()`: Uses `pydub` to get channel count.**

**Processing Flow (Simplified):**
Input -> `app.py` -> `process_audio` -> Get Media Info -> Video Check/Extract -> Normalize -> [Vocal Separation] -> **Check Channels & `split_stereo` flag** -> **[If Stereo Split: Split L/R -> Process L (Diarize+Slice+Transcribe) -> Process R (Diarize+Slice+Transcribe)] OR [If Mono/No Split: Process Mono (Diarize+Slice+Transcribe -> Optional 2nd Pass)]** -> [**Sound Detection (CLAP)** on `no_vocals`] -> **Build YAML structure (Metadata + L/R Segments OR Mono Segments w/ Refinements + Sound Events)** -> Write `master_transcript.yaml` -> Zip -> Cleanup.

**Data Management:**
*   Timestamped output directory.
*   Handles audio/video.
*   **Optional `stereo_split` subdirectory with L/R mono files.**
*   **Output is `master_transcript.yaml` (structured data with metadata).**
*   Conditional `_words`, `2nd_pass` dirs.
*   `zip_results` includes `.yaml` and excludes temp/download/split dirs.
*   `processing.log` file included in output and zip.

**Error Handling:**
*   Includes checks for `ffmpeg`/`ffprobe`.
*   YAML writing has JSON fallback.
*   Error details captured in `final_yaml_data['error']` and written to `master_transcript_error.yaml` on fatal error. 