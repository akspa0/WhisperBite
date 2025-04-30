# Active Context

## Current Focus

**Finalizing `app.py` UI for Two-Pass Workflow:** The core logic for the two-pass ("Event-Guided") preset is implemented in `whisperBite.py`, and the standard preset logic has been restored conditionally. The immediate task is to correctly modify `app.py` to:
1. Add the Gradio Textbox (`pass1_event_prompts_textbox`) specifically for overriding Pass 1 event prompts when the "Event-Guided" preset is selected.
2. Update the `update_ui_for_preset` function to manage the visibility and placeholder text of this new Textbox based on the `preset_dropdown` selection.
3. Update the `run_pipeline` function signature to accept the input from the new Textbox.
4. Add logic within `run_pipeline` to parse the override prompts and pass them correctly (via `preset_kwargs`) to the `get_event_guided_preset` function *only* when that preset is active and the override input is valid.
5. Ensure the `preset_dropdown.change` and `run_button.click` handlers include the new Textbox in their `outputs` and `inputs` lists, respectively.

Previous attempts to apply these `app.py` changes via large edits failed, requiring smaller, targeted edits in the next steps.

## Recent Changes

*   **CLAP Refactoring Completed:** Replaced non-functional `EventDetector` class with `run_clap_event_detection` function.
*   **Two-Pass Workflow Implemented:** Added logic to `whisperBite.py::process_audio` to:
    *   Run Pass 1 CLAP detection.
    *   Call `sound_detection.cut_audio_between_events` based on Pass 1 results.
    *   Loop through resulting segments, performing Pass 2 CLAP annotation, transcription, and soundbite extraction (`sound_detection.extract_soundbites`).
    *   Updated results structure (`results["segments"]`).
*   **Standard Workflow Restored:** Re-added conditional logic in `whisperBite.py::process_audio` to handle standard, non-segment-based presets (`elif not workflow.get("cut_between_events")`).
*   **Preset Structure Fixed:** Corrected `presets.py` functions (e.g., `get_event_guided_preset`) to consistently wrap configurations under the `"config"` key, resolving the `KeyError: 'config'` in `app.py`.
*   **Logger Fix:** Moved `logger = logging.getLogger(__name__)` definition to the top of `sound_detection.py`, resolving the `NameError: name 'logger' is not defined`.

## Active Decisions

*   The two-pass workflow (`Event-Guided` preset) uses separate CLAP configurations for Pass 1 (cutting boundaries, `event_detection` config key) and Pass 2 (segment annotation, `sound_detection` config key).
*   Standard presets reuse the original whole-file processing logic blocks within `whisperBite.py`.
*   `app.py` needs targeted edits to integrate the UI for the Pass 1 prompt override.

## Next Steps

1.  **Fix `app.py` (Targeted Edits):**
    *   Define `pass1_event_prompts_textbox` in the UI layout.
    *   Update `update_ui_for_preset` function.
    *   Update `preset_dropdown.change` handler outputs.
    *   Update `run_pipeline` function signature and internal logic.
    *   Update `run_button.click` handler inputs.
2.  **Test UI Interactivity:** Verify Pass 1 override box visibility.
3.  **Test Workflows:** Run both "Event-Guided" (with/without override) and "Standard" presets.
4.  **Review Outputs:** Check logs and output files (`conversation_segments/`, `soundbites/`, `master_transcript.txt`, `results.yaml`) for correctness.

## Current Focus

**Debugging Performance Bottleneck:** Investigating extreme slowness (potential stall) during the CLAP Event Detection phase. The process shows 100% CPU usage and 0% GPU usage during the `Detecting events` progress loop, despite logs indicating `cuda` as the target device.

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

*   **UI Refactoring:** Created dedicated "Detection" tab with separate controls for Event and Sound detection, linked interactivity to selected Preset.
*   **Output Structure Restoration:** Implemented unique, timestamped output directory creation per run (`<output_folder>/<input_name>_<timestamp>/`).
*   **Configuration Flow Rework:** Aligned UI (`app.py`), presets (`presets.py`), and backend (`whisperBite.py`) to handle distinct settings for Event Detection and Sound Detection. Filtered empty strings from prompt inputs.
*   **File Logging:** Implemented per-run logging to `processing.log` within the unique output directory.
*   **Debugging Added:**
    *   Added logging to show parameters passed to detection functions (`detect_and_save_events`, `detect_sound_events`).
    *   Added logging inside NMS (`apply_temporal_nms`).
    *   Added logging to check PyTorch device for model/tensors before CLAP inference.
    *   Added timing logs around audio loading/resampling (`sf.read`, `librosa.load`, `librosa.resample`).
*   **Previous Fixes:** `stop_event` AttributeError, `KeyError: 'workflow'`, `KeyError: 'name'`, `TypeError` in event-guided transcription, initial prompt logic fixes.

## Active Decisions

*   Focus on resolving the CPU/GPU performance bottleneck before further feature work or workflow refactoring.
*   Event/Sound detection have separate UI controls and configuration paths.
*   Output is organized into unique, timestamped run directories with subfolders.

## Next Steps

1.  **System Reboot (User Action):** User to restart WSL (`wsl --shutdown`) and/or the entire machine to rule out transient system/driver issues affecting GPU utilization.
2.  **Re-run & Check Logs:** After reboot, run again (e.g., "Event-Guided" preset, 0.5 threshold, empty prompts).
3.  **Analyze `processing.log`:**
    *   Verify device logs (`[CHUNKPROC] Model device:`, `[SOUNDPROC] Model device:`, etc.) confirm `cuda` during inference steps.
    *   Note the time taken for loading/resampling steps.
    *   Observe if the `Detecting events` progress bar moves and if GPU utilization increases during this phase.
4.  **If Stall Persists:** Further investigation is needed, potentially focusing on:
    *   PyTorch/CUDA environment validation (versions, compatibility).
    *   Adding more granular logging *within* the CLAP model inference calls (`