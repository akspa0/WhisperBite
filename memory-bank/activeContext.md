# Active Context

## Current Work Focus

- **Implement Speech Transcription Feature:**
  - Create `core/transcription.py` module.
  - Implement `initialize_whisper_model` to load `openai/whisper-large-v3`.
  - Implement `transcribe_segment` for transcribing individual audio segments, including logic for word-level timestamps.
  - Update `audiosegmenter_cli.py` to integrate the transcription workflow:
    - Add new CLI options for transcription (enable/disable, model choice, language, word timestamps).
    - Call transcription functions for each speaker segment.
    - Adjust timestamps to be global.
    - Save individual and reconstructed full transcripts.
- Refine and test existing refactored components (`diarization.py`, `clap_annotator.py`, `audio_utils.py`) to ensure they align with `projectbrief.md` and integrate with the new transcription feature.

## Recent Changes

- **Refactored Core Modules:**
  - `core/diarization.py` updated to use `pyannote/speaker-diarization-3.1` and new segmentation logic.
  - `core/clap_annotator.py` updated to use `microsoft/clap-htsat-unfused` as default.
  - `core/audio_utils.py` updated to correctly pass `target_lufs` for normalization.
  - `audiosegmenter_cli.py` updated to fix import errors, parameter passing for diarization and normalization.
- **Memory Bank Updated (Initial Pass):** `projectbrief.md`, `activeContext.md`, `progress.md` updated to reflect refactoring work.
- Identified discrepancies between original implementation and `projectbrief.md`.
- Initial project setup and documentation (`projectbrief.md`).

## Next Steps (High-Level for this phase)

1.  **Develop `core/transcription.py` module.**
2.  **Integrate transcription into `audiosegmenter_cli.py`.**
3.  **Thoroughly test the complete pipeline including diarization, (optional) CLAP annotation, and transcription.**
4.  Update `results_summary.json` generation to include transcription outputs.
5.  Refine error handling and logging for all components.
6.  Update `README.md` with new features and CLI options. 