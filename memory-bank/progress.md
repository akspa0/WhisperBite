# Progress

**What Works:**
*   Main processing pipeline (normalization, optional vocal separation, diarization, slicing, transcription).
*   Gradio Web UI (`app.py`).
*   Command-line interface (`whisperBite.py`).
*   Handling of file (audio/video), directory (newest file), and URL inputs.
*   Automatic audio extraction from video files using `ffmpeg`.
*   Basic error handling and logging.
*   Manual speaker count setting (now default).
*   Optional automatic speaker count detection (disabled by default).
*   Optional second-pass diarization refinement logic implemented (mono path).
*   Optional word audio extraction (disabled by default).
*   **Optional Sound Detection using CLAP:** 
    *   Implemented via `sound_detection.py` using `transformers`.
    *   Requires `--enable_vocal_separation`.
    *   Configurable via CLI (`--attempt_sound_detection`, `--clap_chunk_duration`, `--clap_threshold`) and UI.
*   Result zipping.
*   Correct Demucs execution (using `-n` flag and internal chunking).
*   Speaker labels formatted as `S0`, `S1`, `S0_L`, `S0_R`.
*   YAML Output (`master_transcript.yaml`) with metadata and structured segments (mono/stereo/second-pass/sound events).
*   Configurable Second Pass (`--second_pass_min_duration`).
*   Optional Stereo Splitting (`--split_stereo`).
*   Dependency management via `requirements.txt` (updated for CLAP).
*   File logging to output directory (`processing.log`).

**What Needs Work/Verification:**
*   **Implement CLAP - Configurable Prompts:** Add UI/CLI options (`--clap_target_prompts`) for custom sound prompts.
*   **Implement CLAP - Event Logging:** Log detected sound events (label, time, confidence) to `processing.log`.
*   **Implement UI - Pyannote Clarity:** Add info text to UI explaining Pyannote/HF token need for diarization.
*   **Testing (Post-Enhancements):** 
    *   CLAP Accuracy/Performance (default/custom prompts).
    *   CLAP Event Logging verification.
    *   Stereo Split Functionality & interactions.
    *   Second Pass (Mono) Merging Logic (check logs).
    *   YAML Output Structure & relative paths.
*   **Implement Second Pass (Stereo):** Add refinement logic for the `split_stereo` path.
*   **Implement Auto Speaker Detection (Stereo):** Add per-channel detection logic.
*   Evaluate Robustness: `detect_optimal_speakers` (mono), media info extraction.
*   Evaluate Edge Cases: File handling, tool failures.
*   Performance Optimization: Long files (Whisper, Pyannote, Demucs).
*   Tune Parameters: Demucs chunking, audio fades/merge gaps.
*   Improve Dependency Checking: `ffmpeg`/`demucs` availability/version.
*   Address Gradio temporary file cleanup.
*   Packaging for deployment.

**Current Status:** CLAP sound detection implemented, replacing YAMNet. `requirements.txt` updated. Configuration options added to CLI and UI. **Planning CLAP enhancements (prompts, logging) and UI clarity updates before testing.**

**Known Issues:**
*   Potential duplication in master transcript when second pass is used (mono path - needs log verification).
*   Second pass refinement **only implemented for mono processing path.**
*   Auto speaker detection **not implemented per-channel for stereo split.**
*   Folder input only processes the newest compatible file.
*   Potential filename parsing errors in `transcribe_with_whisper` (fallback exists).
*   Demucs availability check is basic.
*   Sound detection accuracy depends heavily on CLAP model, prompts, threshold, and audio quality.