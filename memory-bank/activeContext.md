# Active Context

## Current Focus

**Refactoring CLAP Event Detection:** The CLAP Event Detection step (`EventDetector` class in `event_detection.py`) consistently hangs at the start of processing the first audio chunk. This occurs despite correct dependency installation (PyTorch w/ CUDA), successful model loading onto the GPU, and successful text feature precomputation. CPU usage maxes out while the GPU remains idle, indicating a deadlock or fundamental issue with the current implementation's handling of the first CUDA operation on audio data. Previous debugging attempts (forcing CPU, disabling autocast, adding granular logging) did not resolve the hang.

## Recent Changes

*   **Dependencies:** Installed latest PyTorch (2.7.0) with CUDA support in a clean conda environment. Installed all other requirements (including fixing missing `psutil`).
*   **Debugging Attempts (Event Detection):**
    *   Verified CUDA detection and model loading logs.
    *   Forced CPU temporarily (still hung, confirming issue wasn't solely CUDA execution).
    *   Disabled/Re-enabled `torch.amp.autocast`.
    *   Added granular `[CHUNKPROC]` logging and timing within `process_audio_chunk`. -> Confirmed hang occurs before/during the first call to this function or immediately upon entering it.
*   **(Previous)** UI Refactoring (Detection Tab), Output Structure Restoration, Config Flow Rework, File Logging, Various Fixes.

## Active Decisions

*   **Abandon `EventDetector` Fixes:** Stop trying to patch the existing `EventDetector` class implementation due to the persistent, unexplained hang.
*   **Adopt Refactoring Plan:** Proceed with the plan below to rebuild the event detection logic.

## Refactoring Plan: CLAP Event Detection

1.  **Centralize CLAP Model Loading:**
    *   Modify `whisperBite.py::process_audio` to load the CLAP model (`ClapModel`) and processor (`ClapProcessor`) *once* at the beginning if the `"detect_events"` step is enabled in the preset's workflow.
    *   This central loading point will manage the device selection (`cuda`/`cpu`) and keep the loaded model/processor objects available.

2.  **Simplify Event Detection Logic:**
    *   Remove the `EventDetector` class from `event_detection.py`.
    *   Create a new, straightforward function within `event_detection.py`, named `run_clap_event_detection(audio_data, sample_rate, clap_model, clap_processor, device, target_events, threshold, chunk_duration, min_gap)`.
    *   This function will take the pre-loaded model, processor, audio data (as a NumPy array), device, and configuration parameters as input.
    *   It will contain the core logic:
        *   Precompute text features using the passed processor/model/device.
        *   Iterate through audio chunks.
        *   Prepare inputs using the processor.
        *   Run inference (`clap_model.get_audio_features`) within `torch.no_grad()` and `torch.amp.autocast` contexts.
        *   Calculate similarity.
        *   Collect detections.
        *   Apply Temporal NMS (using the existing static method, perhaps moved outside the old class).

3.  **Update `whisperBite.py::process_audio`:**
    *   Import the new `run_clap_event_detection` function.
    *   Add the CLAP model/processor loading logic mentioned in step 1.
    *   Read event detection parameters (`target_events`, `threshold`, etc.) from the `preset_config["event_detection"]` dictionary. Provide sensible defaults if keys are missing.
    *   Load the audio file into a NumPy array using `soundfile.read` (or similar) *before* calling the detection function.
    *   If event detection is enabled, call `run_clap_event_detection`, passing all required arguments.
    *   Handle saving the returned event dictionary to `events/events.json`.

4.  **Refine Logging:**
    *   Keep INFO level logs for model loading, text precomputation, audio loading, and saving results.
    *   Change the detailed intra-chunk logs (`[CHUNKPROC]`) to DEBUG level for optional deep debugging.

## Next Steps

1.  **Implement Refactoring Plan:** Execute the steps outlined above to refactor CLAP event detection.
2.  **Test Refactored Implementation:** Run with the "Event-Guided" preset and verify that event detection proceeds correctly on the GPU without hanging.
3.  **Continue Workflow:** Once event detection is functional, proceed with testing subsequent steps (event-guided transcription, etc.).

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