# Product Context: WhisperBite (2025 Modular Refactor)

## Why This Project Exists
- Audio content is often inaccessible or hard to analyze, especially for multi-speaker or event-rich recordings.
- Monolithic codebases lead to regressions, slow development, and headaches.
- Modular, multi-file architecture enables maintainability, extensibility, and reliability.

## Problems Solved
- Modular, rule-driven workflows for audio processing, supporting both simple and advanced use cases.
- Unified outputs: speaker-attributed, context-annotated transcripts and metadata, organized for downstream AI or human use.
- Workflow sharing: Users can share and reproduce processing pipelines via `config.yaml`.
- Each module is small, focused, and independently testable, reducing the risk of regressions.

## User Experience Goals
- Intuitive workflow authoring (YAML, UI planned).
- Easy sharing and reproducibility of workflows.
- Outputs are accessible, richly annotated, and easy to search or analyze.
- Developers can extend or fix one module without breaking others.

## Use Cases
- Call center analysis, podcast transcription, content management, research, accessibility, and more.

## Path Forward
- Scaffold and migrate to a modular, multi-file codebase.
- Implement and validate the CLAP-driven segmentation workflow as the baseline.
- Expand documentation and workflow examples.
- Prepare for UI-based workflow editing and LLM-driven orchestration.

## Success Metrics
- Output accuracy and completeness.
- Workflow reproducibility and shareability.
- User satisfaction with modularity and extensibility.

## How It Should Work
- User provides an audio file (podcast, call, show, etc.).
- The tool separates vocals, diarizes speakers, transcribes each segment, and annotates the timeline with events (music, laughter, etc.).
- Output is a structured, speaker-attributed, context-annotated transcript and metadata.
- **Canonical pipeline order:** Input & normalization → Demucs → Diarization → Transcription → Soundbite extraction → CLAP annotation (context only) → Output writing.
- **Robust output structure:** Always produce master transcript YAML, per-segment TXT files, and soundbites, matching legacy output format.

## Purpose
WhisperBite addresses the need for sophisticated audio processing in various domains:
1. **Call Analysis**
   - Automated call segmentation
   - Call content transcription
   - Call event detection

2. **Audio Processing**
   - Multi-track handling
   - Vocal separation
   - Sound event detection

3. **Content Organization**
   - Call-based segmentation
   - Structured metadata
   - Multi-track preservation

## Problem Space

### Core Problems Solved
1. **Call Processing Complexity**
   - Manual call separation
   - Call boundary identification
   - Multi-track coordination

2. **Audio Analysis Challenges**
   - Track synchronization
   - Timing preservation
   - Quality optimization

3. **Content Organization**
   - Call-based structuring
   - Track relationship management
   - Metadata organization

### User Experience Goals

1. **Efficiency**
   - Automated call detection
   - Streamlined processing
   - Organized output

2. **Accuracy**
   - Precise call boundaries
   - Clean vocal processing
   - Accurate transcription

3. **Flexibility**
   - Optional track preservation
   - Configurable processing via presets
   - **Separate configuration for Event/Sound detection**
   - Customizable output (YAML structure)

4. **Usability & Traceability**
- Clear workflow via UI Tabs (Input, Processing, Detection)
- Intuitive organization of options
- Accessible results through organized outputs
- **Detailed, traceable output structure:** Unique timestamped run directory with subfolders for each processing step (`normalized/`, `demucs/`, `events/`, `sounds/`, etc.).

**Problem:** Analyzing spoken audio, especially conversations with multiple speakers, is time-consuming. **Getting audio from video files requires extra steps.** Manually transcribing, identifying who said what, and finding specific words is inefficient. Furthermore, automated speaker diarization isn't always perfect and can sometimes group speech from multiple speakers into a single segment.

**Solution:** WhisperBite automates this process via a user-friendly web interface (`app.py`) or a command-line tool (`whisperBite.py`). It accepts **audio or video files**, URLs, or folders (processing the newest compatible file within). **For video inputs, it automatically extracts the audio** before performing diarization and transcription.
*   **Refinement:** It includes an **optional second pass** that re-analyzes longer segments from the first pass to detect and separate missed speaker changes, improving accuracy.
*   **Word Extraction:** It can optionally isolate individual words with their timings and audio snippets, though this feature is **disabled by default** as it generates many files.
*   **Sound Detection:** If vocal separation is enabled, it can **optionally attempt** to identify non-speech sounds (like music or noise tags from Whisper) in the non-vocal audio track and include them in the master transcript.

**How it Works (User Perspective - Web UI):**
1.  Launch the Gradio app (`python app.py`).
2.  Provide source (upload audio/video file, specify folder path [processes newest file], or enter URL).
3.  Configure options: Whisper model, speaker settings (manual count [default] or auto-detect), vocal separation.
4.  **Configure advanced options:** Enable second pass refinement? Enable word audio extraction? Enable sound detection (if vocal separation is on)?
5.  Provide Hugging Face token (for diarization).
6.  Specify output directory.
7.  Click "Process Audio". (The app extracts audio from video automatically if needed).
8.  Status updates appear. If second pass is enabled, this stage will take longer.
9.  Upon completion, a transcript preview (preferring the refined transcript if second pass was run) is shown, and a download link for the results zip file is provided.

**User Experience Goals:**
*   Provide an intuitive web UI for easy configuration and execution.
*   **Support direct processing of video files by handling audio extraction.**
*   Offer a fallback command-line interface for advanced use cases.
*   Automate complex audio processing tasks.
*   **Address common diarization inaccuracies with an optional refinement step.**
*   Generate detailed and usable outputs.
*   **Allow users to control the verbosity of output (e.g., word extraction, sound detection).**
*   Offer flexibility through optional processing steps and model choices.

## Recent Regression & Lessons
- A bug in the VAD output format (tuples vs dicts) broke CLAP, diarization chunking, and all downstream outputs.
- **Lesson:** All pipeline utility functions must return type-consistent, schema-validated outputs (e.g., dicts with 'start'/'end' keys).
- Output parity with the legacy implementation is required before further modularization or feature work. 