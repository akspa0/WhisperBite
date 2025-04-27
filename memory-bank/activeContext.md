# Active Context

**Current Work Focus:** Refining output generation and adding sound detection.

**Recent Changes:**
*   Refactored Demucs integration:
    *   Removed faulty `--version` check.
    *   Simplified path handling in `vocal_separation.py`.
    *   Corrected command flag from `--model` to `-n`.
    *   `separate_vocals_with_demucs` now returns paths for both `vocals` and `no_vocals` tracks.
*   Implemented Speaker Label Formatting:
    *   Added `format_speaker_label` helper.
    *   Updated slicing, transcription, and refinement functions to use `S0`, `S1` format for outputs.
*   Refactored Master Transcript Generation:
    *   Removed transcript writing from `transcribe_with_whisper` and `run_second_pass_diarization`.
    *   Implemented merging logic in `process_audio` to combine first-pass and second-pass results, attempting to replace originals with refined segments.
    *   Added sorting and final writing of `master_transcript.txt` within `process_audio`.
*   Added Optional Sound Detection:
    *   Added UI checkbox in `app.py`, enabled only when vocal separation is on.
    *   Passed option to `process_audio`.
    *   If enabled, `process_audio` transcribes the `no_vocals.wav` track.
    *   Uses regex to identify bracketed/parenthesized tags (e.g., `[ music ]`, `(noise)`) as sound events.
    *   Adds these events (labeled `SOUND`) to the final segment list before sorting and writing the master transcript.
*   Changed Auto Speaker Detection Default:
    *   Set default to `False` in UI (`app.py`) and backend (`process_audio`).
*   Updated `README.md` with new features and notes.
*   Added detailed logging for transcript merging to aid debugging potential duplication issues.
*   Added `--segment` and `--overlap` flags to the Demucs command in `vocal_separation.py` to enable internal chunking for large files, preventing potential memory errors. **Values adjusted to `--segment 7` (due to htdemucs limit) and `--overlap 0.25` (default).**
*   Created `sound_detection.py` with YAMNet integration:
    *   **Added new dependencies (`tensorflow`, `tensorflow_hub`, `librosa`).**
    *   **Implemented `detect_sound_events` function using YAMNet model from TF Hub.**
    *   **Targets specific sound classes (phone ringing, dial tone etc.) based on a threshold.**
    *   **Includes basic segment merging for consecutive detections.**
*   **Modified `process_audio` in `whisperBite.py`:**
    *   **Imports and calls `detect_sound_events` for the `no_vocals` track (if available/enabled).**
    *   **Replaced previous regex-based sound detection.**

**Next Steps / Potential Issues:**
*   **Refine YAMNet Class Map Loading:** The current hardcoded dictionary in `sound_detection.py` should be replaced with loading the actual `yamnet_class_map.csv` file (either fetch from TF Hub cache or include in repo).
*   **Test YAMNet Sound Detection:** Thoroughly test with audio containing target sounds (phone ringing, dial tones) and other noises to evaluate accuracy, adjust threshold, and refine segment merging if needed.
*   Test Demucs chunking: Verify that the `--segment 7` and `--overlap 0.25` flags work correctly for large files (>1 hour) and that the output quality is acceptable, especially at chunk boundaries. Adjust values if necessary.
*   Verify the master transcript merging logic thoroughly to ensure duplicates are correctly eliminated when the second pass is used. (Initial user feedback suggests this might still need work). Logs were added to help diagnose.
*   Evaluate the effectiveness of sound detection using Whisper on the `no_vocals` track. Consider alternative methods (e.g., dedicated SED models like YAMNet) if Whisper is insufficient.
*   Test phone ringing detection specifically.
*   Package application for multi-platform deployment (e.g., using Pinokio). 