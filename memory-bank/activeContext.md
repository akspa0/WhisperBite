# Active Context

## Current Focus

1.  **UI Refactoring & Clarity:**
    *   Create a dedicated "Detection" tab in the Gradio UI.
    *   Implement separate input controls within the "Detection" tab for *Event Detection* (prompts, threshold, min gap) and *Sound Detection (CLAP)* (prompts, threshold, chunk duration).
    *   Link the interactivity (enabled/disabled state) of these new controls to the selected *Preset*, activating only the relevant controls based on the preset's workflow (`detect_events` or `detect_sounds`).
    *   Remove redundant/confusing controls (orphaned CLAP toggle, duplicate sound detection checkboxes).
2.  **Output Structure Restoration:**
    *   Reinstate the original output behavior: creating a unique, timestamped root directory for each processing run, named after the input file (e.g., `<output_folder>/<input_name>_<timestamp>/`).
    *   Ensure all intermediate and final files are saved into step-specific subdirectories within this unique run folder (e.g., `normalized/`, `demucs/`, `events/`, `sounds/`, `speakers/`, `transcriptions/`).
3.  **Backend Logic Alignment:**
    *   Update `app.py` (`process_wrapper`) to gather inputs from the new UI structure.
    *   Update `presets.py` to accept and utilize distinct kwargs for event detection and sound detection settings.
    *   Ensure `whisperBite.py` uses the correct configuration keys and defaults passed from the updated presets for both detection types.

## Recent Changes

*   **Fixed `stop_event` Handling:** Resolved `AttributeError` by passing a `threading.Event` from `app.py` to `whisperBite.py` instead of a lambda function. Modified `stop_current_job` to set the event.
*   **Fixed `KeyError: 'workflow'`:** Refactored `presets.py` to move the `workflow` configuration (as a dict of booleans) inside the `config` dictionary, matching the structure expected by `process_audio`.
*   **Fixed `KeyError: 'name'`:** Modified `process_audio` to accept `preset_name` as a direct argument and updated the call site in `app.py`.
*   **Fixed CLAP Prompt Issues (Initial):** Corrected logic in `whisperBite.py` (`process_audio`) for *event detection* to handle potentially empty prompt lists from config and use `DEFAULT_EVENTS`. Corrected key usage (`min_duration` vs `min_gap`).
*   **Fixed `TypeError` in Event-Guided Transcription:** Corrected the loop in `whisperBite.py` to iterate over the list of detection dictionaries within `results["events"]` and use `event["type"]` instead of `event["label"]`. Added fallback and error handling.
*   **Partially Fixed CLAP Prompts:** Updated `presets.py` to use correct kwargs (`clap_target_prompts`, etc.) for *sound detection* config. Updated `whisperBite.py` to use correct keys/defaults for sound detection. *(Self-correction: Realized this didn't fix prompts for EVENT detection, leading to current plan).*

## Active Decisions

*   Event Detection and Sound Detection (CLAP) require separate configuration controls in the UI for clarity.
*   The processing pipeline *must* generate a unique, timestamped output directory per run, containing standardized subdirectories for each processing stage's outputs.
*   UI control interactivity should be driven by the selected Preset to guide the user.

## Next Steps

1.  **Implement UI Refactoring (`app.py`):** Create "Detection" tab, move/add controls, remove old ones, update interactivity based on presets.
2.  **Implement Output Directory Structure (`app.py`):** Modify `process_wrapper` to create the timestamped root directory and pass its path to `run_pipeline`. Modify `run_pipeline`/`process_audio` (or helpers) to use this path and create necessary subdirectories.
3.  **Update Backend (`app.py`, `presets.py`, `whisperBite.py`):** Adapt `process_wrapper` to handle new UI inputs, modify `presets.py` to use new distinct kwargs, verify `whisperBite.py` uses correct keys from config.
4.  **Testing:** Thoroughly test different presets, custom prompts for both detection types, and verify the output directory structure and file contents.

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
    *   *(Low Priority)* **DAC Encoding:** Consider adding option to encode output audio snippets using DAC model (`transformers.DacModel`) for high-efficiency storage.
*   Package application (Pinokio). 