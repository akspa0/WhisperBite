# Active Context

**Current Work Focus:** **Implementing planned CLAP enhancements (configurable prompts, event logging) and UI clarity updates before testing.**

**Recent Changes:**
*   **Refactored Sound Detection to use CLAP:**
    *   Replaced YAMNet/TensorFlow with `transformers.ClapModel`.
    *   Added basic CLI/UI config for enable/chunk/threshold.
    *   Integrated into `process_audio` and YAML output.
*   **Updated Dependencies:** Removed TF, added `transformers`.
*   Refactored Demucs integration (paths, flags, chunking).
*   Refactored Master Transcript Generation (YAML output, merging logic for mono second pass).
*   Implemented Speaker Label Formatting (`S0`, `S1`, `S0_L`, `S0_R`).
*   Added Stereo Splitting (`--split_stereo`).
*   Added Media Info extraction.
*   Added Second Pass configuration (min duration).
*   Added detailed logging for transcript merging debugging.
*   Set auto speaker detection default to `False`.

**Next Steps / Planned Enhancements:**
*   **CLAP - Configurable Target Prompts:** 
    *   Add UI Textbox (`app.py`) and CLI arg (`--clap_target_prompts` in `whisperBite.py`) for comma-separated prompts.
    *   Update `process_audio` to parse and pass prompts to `detect_sound_events`.
    *   Add used prompts to YAML metadata.
*   **CLAP - Log Detected Events:** 
    *   Modify `process_audio` to loop through `sound_event_segments` and log each detected event (label, time, confidence) to `processing.log`.
*   **UI Clarity - Pyannote/HF Token:**
    *   Add info text to speaker count slider/checkbox in `app.py` explaining Pyannote dependency and HF token requirement for diarization.
*   **Testing (After Enhancements):**
    *   CLAP Accuracy/Performance (with default and custom prompts).
    *   Stereo Split Functionality & interactions.
    *   YAML Output Structure & relative paths.
    *   Second Pass Merging (Mono) - check logs for duplication.
    *   Demucs Chunking stability/quality.
*   **Known Issues / Future Refinements:**
    *   Second-pass refinement **not implemented** for `split_stereo` path.
    *   Auto speaker detection **not implemented per-channel** for `split_stereo` path.
*   Package application (Pinokio). 