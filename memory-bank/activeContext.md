# Active Context: Canonical Workflow & Vision (as of [today's date])

## Canonical Workflow
1. **Input & Normalization**
   - Input audio is normalized for consistent loudness.
2. **Demucs Vocal Separation**
   - Demucs splits audio into vocals (for speech) and accompaniment.
3. **Speaker Diarization (pyannote)**
   - Run on Demucs vocal track for best accuracy.
   - Output: time-stamped, speaker-labeled segments.
4. **Whisper Transcription**
   - Run on each diarized segment (Demucs vocals).
   - Output: per-speaker, time-aligned transcripts.
5. **Soundbite Extraction**
   - Extract per-speaker/per-segment soundbites from Demucs vocals.
6. **Call/Conversation Reconstruction**
   - Use normalized track for context-rich call grouping (future step).
7. **CLAP Event Annotation**
   - Run on normalized track (full context).
   - Output: timeline of events (music, laughter, etc.)
8. **Contextual Transcript/Output Writing**
   - Combine all data: speaker, transcript, event context.
   - Output: canonical, accessible, richly annotated transcript and metadata.

## Vision
- WhisperBite augments audio with speaker-based transcriptions and contextual hints.
- Output is designed for accessibility (e.g., for the deaf) and for downstream AI (LLMs, image generation) to reconstruct or illustrate scenes.

## Next Steps
- Ensure pipeline and workflow YAML route the correct audio to each step.
- Restore/implement per-speaker segment extraction and transcript rebuilding as in the legacy pipeline.
- Validate output structure and content against canonical examples.

## Current Focus

**CRITICAL REGRESSION: Modular Refactor Broke Core Output Features**

- As of the latest modular pipeline refactor, **no transcripts are being written at all** (neither per-segment nor master transcript YAML).
- **CLAP event detection is still non-functional**: no events are detected, regardless of input or settings.
- **Soundbite extraction is not producing expected outputs**: soundbites are not being cut or transcribed as in the legacy version.
- The pipeline runs, but the most important outputs (transcripts, soundbites, event segments) are missing or empty.

## Debugging Steps Taken
- Restored robust CLAP and soundbite logic from the old version, including model loading, device handling, and segment extraction.
- Re-implemented `extract_audio_segment` and ensured it is available in `utils.py`.
- Added detailed logging for model loading, device, audio stats, and output files.
- Confirmed that the pipeline steps are being called, but output files are not being written as expected.
- Verified that the modular pipeline is passing through context and parameters, but downstream steps (transcription, soundbite, transcript writing) are not producing files.

## Current Blockers
- **No transcripts are being written**: This is a showstopper regression. The pipeline must always produce a master transcript YAML and per-segment TXT files.
- **CLAP event detection is non-functional**: No events are detected, even on files that worked in the legacy version.
- **Soundbite extraction is not producing .wav or .txt files**: The output folders are created, but files are missing.

## Next Steps (Highest Priority)
1. **Restore transcript writing:** Ensure that the `write_transcripts` step is always called and produces both the master YAML and per-segment TXT files.
2. **Fix CLAP event detection:** Debug why CLAP is not detecting any events, even with known-good audio and prompts. Compare with the old version and test with minimal settings.
3. **Restore soundbite extraction and transcription:** Ensure that soundbites are always cut and transcribed, and outputs are written to disk.
4. **Add regression tests:** Prevent future loss of core output features.
5. Only after these are fixed: resume modular feature expansion and UI work.

## Summary
- The modular pipeline refactor has broken the most critical output features of WhisperBite.
- Immediate focus must be on restoring transcript writing, robust CLAP event detection, and soundbite extraction before any further modularization or UI work.
- All debugging and development should be directed toward fixing these regressions and validating outputs on known-good test files.

## Feature Checklist (to be mapped to modular steps and UI)

### Input Handling
- Single audio file (mono/stereo)
- Multiple audio files (e.g., left/right, us/them)
- Folder input (batch processing)
- Video file input (auto audio extraction)
- URL input (yt-dlp, direct download)
- Support for all common audio/video formats
- Automatic detection of newest file in folder

### Preprocessing
- Audio normalization (LUFS target)
- Optional stereo channel splitting
- Channel assignment (left/right, us/them, custom)
- Batch normalization for multi-file/folder input

### Vocal Separation (Demucs)
- Optional Demucs separation (vocals/no_vocals)
- Model selection
- Per-channel Demucs processing
- Output of both vocal and no_vocal tracks

### Speaker Diarization (Pyannote)
- Diarization on any input or Demucs output
- Automatic speaker count detection
- Manual speaker count override
- Per-channel diarization
- Adjustable merging parameters
- Second pass refinement (optional)
- Output RTTM and YAML with speaker segments

### Transcription (Whisper)
- Model selection
- Language selection/override
- Per-segment transcription (from diarization)
- Per-channel transcription
- Word-level timestamps (optional)
- VAD filtering (optional)
- Output of full transcript and per-segment/word transcripts

### Sound Event Detection (CLAP)
- CLAP event detection on any track
- Customizable prompts
- Adjustable threshold, chunk duration, min gap
- Event-action mapping (e.g., segment on "ringing phone")
- Output of detected events (JSON/YAML)

### Soundbite Extraction
- Per-segment/word soundbite extraction
- Output format selection (WAV, MP3, FLAC)
- Bitrate/compression selection for MP3/FLAC
- Option to force mono (default: off)
- Filename: numerical prefix + transcription (max 128 chars, sanitized)
- Per-speaker and per-channel output folders
- Embedding of metadata (show name, date, speaker, channel, etc.)
- **ASR/VAD-based boundary detection for natural soundbites**

### Output Organization
- Unique, timestamped run directory per input/batch
- Subfolders for each processing step
- Per-speaker and per-channel subfolders for soundbites
- Master transcript YAML with all metadata, lineage, and processing parameters
- Per-run config.yaml and processing.log

### Metadata Handling
- Extraction of all input file metadata
- Logging of all metadata in YAML output
- Embedding of key metadata in soundbite outputs
- Fallback to filename for show name if missing

### UI Features
- Modular workflow builder (add/remove/reorder steps)
- Parameter editor for each step
- Channel assignment and mixing controls (pan, left/right %)
- Soundbite output settings (format, bitrate, mono/stereo)
- Directory and metadata preview
- Batch file/folder preview and selection
- Save/load/export workflows as YAML
- Run workflow on selected input(s)
- Progress/status updates and error reporting

### Advanced/Legacy Workflows
- Support for "mixing" workflows (e.g., preFapMix.py logic)
- Event-driven segmentation and action mapping
- Multi-channel (up to 2) support with custom naming
- Reproducibility: all parameters and steps logged for each run

### Other
- Stop/cancel mechanism
- Output zipping/compression for sharing
- Regression test coverage for all legacy workflows

## Active Decisions

*   All features must be preserved and exposed as modular steps.
*   No hidden/hard-coded logic; all workflow/rule logic is user-driven via the UI.
*   Regression testing and feature checklist are required before removing legacy code.

## Next Steps

1. Implement the modular backend pipeline engine.
2. Remove all hard-coded workflows and presets from the codebase.
3. Implement YAML-driven workflow loading, saving, and execution (workflows/ folder).
4. Validate that all features and legacy workflows can be recreated as YAML workflows.
5. After backend is functional, design and build the new UI to interface with the modular system.
6. Update documentation and memory bank to reflect the new architecture and workflow system.

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

## Modular Pipeline Refactor (Ongoing)

- Demucs runs immediately after normalization (if enabled), producing vocals and no_vocals tracks.
- CLAP Pass 1 runs on normalized track to segment calls/conversations using user prompts.
- For each call segment:
    - Diarization (manual speaker count if provided) on vocals track if available, else normalized.
    - VAD on vocals track if available, else normalized, to refine soundbite boundaries.
    - Whisper transcription for the full segment (with speaker labels).
    - After all annotation, cut soundbites using VAD and annotate with speaker/event data, using vocals/no_vocals/remix as per user config.
- CLAP Pass 2 runs on no_vocals for event annotation (user prompts).
- CLAP Pass 3 runs on normalized for additional annotation (user prompts).
- All user options (CLAP prompts, soundbite source, remix volumes) are respected.
- Implementation is ongoing. Next steps: fill in per-segment logic and modular soundbite extraction.

## Modular Pipeline Status (2025-05-04)

- The modular pipeline is now fully operational and tested. All major steps are implemented as modular, pluggable steps:
  - normalize, demucs, clap, diarization, vad, whisper, annotate_segments, soundbite, write_transcripts
- CLI and workflow YAML integration is complete. Users can run any modular workflow from the CLI using the --workflow_yaml argument.
- Demucs output structure is now flat and VRAM-optimized:
  - Uses --segment 7 for VRAM management
  - Uses --filename to flatten output naming
  - No more nested or redundant folders
- Device handling is robust: all steps use torch.device objects, preventing string/type errors in CLAP and diarization.
- CLAP, diarization, and all steps are robust to device/config issues.
- Security: Hugging Face token is never logged, only written to config.yaml.

## Next Steps
- Continue modular feature expansion (e.g., support for multiple vocal splitters, UVR integration)
- Add more workflow YAMLs for different use cases and testing
- UI integration for workflow selection and parameter editing
- Ongoing regression testing and documentation updates

## Updated Segmentation and Processing Strategy (2025-05-04)

- **CLAP is now used for initial segmentation:**
  - Detects 'start' events (e.g., 'speech', 'conversation') and 'end' events (e.g., 'telephone hang-up tones').
  - Audio is cut into logical conversation segments at these boundaries.
- **All downstream processing (Demucs, diarization, transcription, further CLAP passes) operates on these segments.**
- **CLAP prompts are simplified for segmentation, but additional prompts are used for later annotation.**
- **Inputs are always audio recordings (phone calls, with conversations, music, sound effects, animal noises, hangup tones, etc.).**
- **Future:** Consider integrating Ultimate Vocal Remover GUI for improved vocal separation, as Demucs is deprecated.
- **Goal:** Build a versatile automatic editor for phone call audio, preserving speech in small, clear, bite-sized pieces.

## 2025-05-04: Critical Output Regression and CLAP Integration Issues

- Modular pipeline outputs (transcripts, soundbites) do **not** match the @older/evenolder implementation, which is the user's expectation and requirement.
- The main goal of CLAP integration was to automate the creation of input segments for the old WhisperBite pipeline, **not** to change the output format or logic.
- Current regressions:
  - Transcript and soundbite outputs are not in the expected format (sequential, per-speaker, with timestamps, as in the old version).
  - Modular pipeline is not producing the same results as the legacy version.
- User is considering reverting to the old version if this cannot be resolved, as the regressions are blocking their workflow.
- **Immediate priority:** Restore output parity with the old version, using CLAP only for coarse segmentation, and running the old pipeline logic (demucs, diarization, speaker slicing, transcription, output writing) on each segment.
- All further modularization and feature work is blocked until this is fixed.

## Next-Generation Workflow Draft (2025-05-04)

- **CLAP is used only for coarse segmentation** (call/conversation boundaries) and event annotation.
- For each CLAP segment, the original @older/evenolder pipeline is run:
  - Normalize
  - Demucs (vocals/no_vocals)
  - Transcribe full segment (vocals)
  - Diarization (vocals)
  - Slice by speaker (vocals)
  - Transcribe each speaker soundbite (vocals)
  - Write outputs: per-speaker soundbites (WAV), per-speaker transcripts (TXT), full segment transcript (TXT), segment YAML (all metadata)
- Each segment gets a YAML file with all metadata (CLAP events, timings, speakers, transcript/soundbite paths, etc.).
- Master transcript and master index YAML are generated for the run.
- Output structure and transcript/soundbite format must match the old version, with new metadata added.
- No output format drift is allowed; all new features are additive and non-breaking.
- This workflow is the immediate target for implementation and regression testing.

## Finalized Output Structure and Naming Convention (2025-05-04)

- All outputs (audio, transcripts, metadata) use descriptive, human-friendly filenames: a zero-padded sequence prefix plus a short, sanitized transcript snippet (e.g., 0001_hello_this_is_bob.wav).
- Outputs are organized in a clear hierarchy: source/ -> normalized/ -> demucs/ -> vocals/, no_vocals/ -> speakers/ -> soundbites_per_speaker/.
- Each speaker has a folder with all their soundbites and transcripts.
- Each soundbite has a YAML with all metadata (sequence, speaker, timing, transcript, CLAP events, call grouping, file paths).
- A master transcript and master YAML provide a global, sequential view and all metadata.
- This structure allows easy reconstruction, searching, and further processing.
- This is the canonical output structure for all future implementation.

### Detailed Output Structure and Examples

#### Directory Tree Example

```
<output_root>/
  source/
    original_input.wav
  normalized/
    input_normalized.wav
  demucs/
    vocals.wav
    no_vocals.wav
  speakers/
    S0/
      0001_hello_this_is_bob.wav
      0001_hello_this_is_bob.txt
      0001_hello_this_is_bob.yaml
      0002_next_phrase.wav
      ...
    S1/
      0003_hi_this_is_alice.wav
      ...
  soundbites_per_speaker/
    (symlinks or copies to speakers/SX/ files, if needed)
  transcript_pool/
    0001_hello_this_is_bob.txt
    ...
  metadata_pool/
    0001_hello_this_is_bob.yaml
    ...
  master_transcript.txt
  master_transcript.yaml
  master_index.yaml
  processing.log
```

#### Filename Convention
- All files start with a zero-padded sequence prefix (e.g., `0001_`).
- The next part is a sanitized snippet of the transcript (e.g., `hello_this_is_bob`).
- Example: `0001_hello_this_is_bob.wav`, `0001_hello_this_is_bob.txt`, `0001_hello_this_is_bob.yaml`.

#### Example Soundbite YAML Metadata
```yaml
sequence: 1
speaker: S0
start: 12.34
end: 18.56
transcript: "Hello, this is Bob."
clap_events:
  - type: "music"
    start: 18.00
    end: 18.56
call_id: 0
file_paths:
  audio: speakers/S0/0001_hello_this_is_bob.wav
  transcript: speakers/S0/0001_hello_this_is_bob.txt
  yaml: speakers/S0/0001_hello_this_is_bob.yaml
parent_call_audio: demucs/vocals.wav
original_input: source/original_input.wav
```

#### Master Files
- `master_transcript.txt`: Sequential transcript of all soundbites, with speaker, timing, and transcript.
- `master_transcript.yaml`: List of all soundbites and their metadata, in order.
- `master_index.yaml`: Index of all files and their metadata for easy lookup.

#### Canonical Structure
- This structure is canonical and must be followed for all future development.
- All filenames, folders, and metadata fields must match this specification for consistency, searchability, and downstream processing.