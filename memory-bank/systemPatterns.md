# System Patterns

## Architecture Overview

### Core Components
1. **Audio Processing Pipeline**
   ```mermaid
   flowchart TD
       Input[Input File/URL] --> Extract[Audio Extraction]
       Extract --> Normalize[Audio Normalization]
       Normalize --> Original[Normalized Original Track]
       Normalize --> Demucs[Vocal Separation]
       
       Demucs --> NoVocals[Non-Vocal Track]
       Demucs --> Vocals[Vocal Track]
       
       NoVocals --> CallDetect[Call Detection]
       CallDetect --> Boundaries[Call Boundaries]
       
       Boundaries --> CutOriginal[Cut Original Track]
       Boundaries --> CutVocal[Cut Vocal Track]
       Boundaries --> OptionalCutNonVocal[Optional Non-Vocal Cut]
       
       CutVocal --> ProcessCalls[Process Each Call]
       ProcessCalls --> Diarize[Speaker Diarization]
       ProcessCalls --> Detect[Sound Detection]
       ProcessCalls --> Transcribe[Transcription]
   ```

2. **File Structure**
   - Each run generates a unique root directory: `<output_folder>/<input_filename>_<timestamp>/`
   - Subdirectories within the run folder organize outputs by processing step.
   ```mermaid
   graph TD
       Input(Input File/Folder/URL) --> MainOutput{Output Folder}
       MainOutput --> RunDir("<input_name>_<timestamp>")
       
       subgraph Run Directory
           RunDir --> Config(config.yaml)
           RunDir --> Log(processing.log)
           RunDir --> NormDir(normalized/)
           RunDir --> DemucsDir(demucs/)
           RunDir --> EventsDir(events/)
           RunDir --> SoundsDir(sounds/)
           RunDir --> SpeakersDir(speakers/)
           RunDir --> TranscriptsDir(transcriptions/)
           RunDir --> CutsDir(cut_segments/)
           RunDir --> MasterYAML(master_transcript.yaml)
           RunDir --> ResultsZip(results.zip)
       end
       
       NormDir --> NormWav("*.wav")
       DemucsDir --> VocalsWav("vocals.wav")
       DemucsDir --> NoVocalsWav("no_vocals.wav")
       EventsDir --> EventsJSON("events.json")
       SoundsDir --> SoundsJSON("sounds.json")
       SpeakersDir --> SpkSubDir("S0/")
       SpkSubDir --> SpkWav("*.wav")
       TranscriptsDir --> SpkTSDir("S0_transcriptions/")
       SpkTSDir --> SpkTxt("*.txt")
       CutsDir --> CutsWav("*.wav")
       
       style DemucsDir fill:#f9f,stroke:#333,stroke-width:2px, BORDER-STYLE: dashed
       style EventsDir fill:#f9f,stroke:#333,stroke-width:2px, BORDER-STYLE: dashed
       style SoundsDir fill:#f9f,stroke:#333,stroke-width:2px, BORDER-STYLE: dashed
       style CutsDir fill:#f9f,stroke:#333,stroke-width:2px, BORDER-STYLE: dashed
       note right of DemucsDir : Dashed boxes indicate optional outputs dependent on configuration
   ```
   **Subdirectory Purpose:**
   *   `normalized/`: Normalized audio derived from input.
   *   `demucs/`: Output from vocal separation (vocals, no_vocals).
   *   `events/`: Detected event data (e.g., `events.json`).
   *   `sounds/`: Detected sound data (e.g., `sounds.json`).
   *   `speakers/`: Audio segments sliced by speaker (subfolders S0, S1...). Includes `segments.json`.
   *   `transcriptions/`: Text transcriptions per speaker segment (subfolders S0_transcriptions...). Includes full transcripts per speaker.
   *   `cut_segments/`: Audio segments cut based on detected events/sounds.
   *   `master_transcript.yaml`: Structured YAML output containing all segments, metadata, etc.
   *   `config.yaml`: Copy of the preset configuration used for the run.
   *   `processing.log`: Log file for the run.
   *   `results.zip`: Compressed archive of the entire run directory (optional).

## Design Patterns

### 1. Pipeline Pattern
- Call-first processing approach
- Sequential track processing
- Coordinated segment handling
- Synchronized timing management

### 2. Factory Pattern
- Dynamic call segmentation
- Configurable processing components
- Flexible boundary detection

### 3. Strategy Pattern
- Configurable call detection
- Multiple processing modes
- Customizable output formats

### 4. Observer Pattern
- Call boundary monitoring
- Processing progress tracking
- Multi-track synchronization

## Key Technical Decisions

### 1. Call Processing
- CLAP-based boundary detection
- Multi-track cutting coordination
- Segment timing preservation

### 2. Audio Track Management
- Original track preservation
- Vocal track optimization
- Optional non-vocal track storage

### 3. Processing Flow
- Call-centric organization
- Parallel track processing
- Timing synchronization

### 4. Output Structure
- Call-based organization
- Multi-track accessibility
- Comprehensive metadata

## Component Relationships

### 1. Call Detection
- Non-vocal track analysis
- Boundary identification
- Timing coordination

### 2. Track Processing
- Multi-track cutting
- Segment synchronization
- Processing coordination

### 3. Output Generation
- Call-based organization
- Track relationship maintenance
- Metadata compilation

## Error Handling

### 1. Boundary Detection
- Validation checks
- Timing verification
- Overlap management

### 2. Track Processing
- Segment integrity checks
- Processing verification
- Recovery mechanisms

### 3. Output Validation
- Call completeness verification
- Track alignment checks
- Metadata validation

**Overall Architecture:**
*   Core processing pipeline (`whisperBite.py`).
*   Gradio web UI (`app.py`).
*   Conditional Two-Pass Pipeline.

**Key Components/Functions:**
*   `app.py`:
    *   `build_interface()`: Defines UI, accepting audio/video file types.
    *   `run_pipeline()`: Handles UI logic, calls `whisperBite.process_audio`.
*   `whisperBite.py`:
    *   `process_audio()`: Main orchestrator.
        *   Accepts `--split_stereo` flag.
        *   Determines actual input file (handles directory/URL).
        *   Calls `utils.get_media_info`.
        *   Checks if input is video, calls `extract_audio_from_video`.
        *   Calls normalization, optional vocal separation.
        *   **Checks audio channels (`utils.get_audio_channels`) and `split_stereo` flag.**
        *   **If `split_stereo` and 2 channels:**
            *   **Splits audio into L/R mono files.**
            *   **Runs diarization, slicing (with `_L`/`_R` suffix), and transcription independently on L and R channels.**
        *   **Else (mono or `split_stereo` disabled):**
            *   **Runs diarization, slicing, and transcription on the single input track.**
            *   **Conditionally calls `run_second_pass_diarization` (using `second_pass_min_duration`).**
        *   **Conditionally calls `sound_detection.detect_sound_events` on `no_vocals` track.**
        *   **Builds nested Python dictionary containing metadata (including media info, options, URL) and segments (either separate L/R lists or a single list potentially with nested refined segments).**
        *   **Writes the dictionary to `master_transcript.yaml` using `yaml.dump()`.**
        *   Calls `zip_results`.
        *   Cleans up temporary files and file logger.
    *   `extract_audio_from_video()`: Uses `ffmpeg`.
    *   `transcribe_with_whisper()`: Handles transcription, word extraction. **Accepts `speaker_suffix`.**
    *   `run_second_pass_diarization()`: Implements refinement pass. **Accepts `segment_min_duration`.**
    *   `format_speaker_label()`: Converts raw speaker labels.
    *   `slice_audio_by_speaker()`: Slices audio. **Accepts `speaker_suffix`.**
    *   Other helpers.
*   `vocal_separation.py`:
    *   `separate_vocals_with_demucs()`: Runs Demucs with chunking.
*   `sound_detection.py`:
    *   `detect_sound_events()`: **Loads CLAP model/processor. Processes audio in chunks, gets audio embeddings, compares to text prompt embeddings for target sounds, and returns detected event times/labels.**
*   `utils.py`: 
    *   Helpers (`sanitize_filename`, `download_audio`, `zip_results`).
    *   **`get_media_info()`: Uses `ffprobe` to get media metadata.**
    *   **`get_audio_channels()`: Uses `pydub` to get channel count.**

**Processing Flow (Simplified):**
Input -> `app.py` -> `process_audio` -> Get Media Info -> Video Check/Extract -> Normalize -> [Vocal Separation] -> **Check Channels & `split_stereo` flag** -> **[If Stereo Split: Split L/R -> Process L (Diarize+Slice+Transcribe) -> Process R (Diarize+Slice+Transcribe)] OR [If Mono/No Split: Process Mono (Diarize+Slice+Transcribe -> Optional 2nd Pass)]** -> [**Sound Detection (CLAP)** on `no_vocals`] -> **Build YAML structure (Metadata + L/R Segments OR Mono Segments w/ Refinements + Sound Events)** -> Write `master_transcript.yaml` -> Zip -> Cleanup.

**Data Management:**
*   Timestamped output directory.
*   Handles audio/video.
*   **Optional `stereo_split` subdirectory with L/R mono files.**
*   **Output is `master_transcript.yaml` (structured data with metadata).**
*   Conditional `