# Product Context

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

## Use Cases

### 1. Call Center Analysis
- Call segmentation
- Content transcription
- Event detection

### 2. Audio Processing
- Track separation
- Content analysis
- Quality optimization

### 3. Content Management
- Call organization
- Track preservation
- Metadata management

## Success Metrics

### 1. Processing Accuracy
- Call boundary precision
- Transcription quality
- Event detection accuracy

### 2. Efficiency
- Processing speed
- Resource utilization
- Output organization

### 3. Usability
- Workflow clarity
- Result accessibility
- Configuration ease

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