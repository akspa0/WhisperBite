# System Patterns

## Architecture Overview

### Modular, User-Driven Workflow Architecture (Planned)
- All processing steps (input, normalization, Demucs, Pyannote diarization, Whisper transcription, CLAP, soundbite extraction, etc.) are modular and can be chained in any order via a user-driven workflow editor in the UI.
- The backend dynamically constructs and executes the pipeline based on the user's workflow definition (from the UI or YAML).
- All legacy features are preserved as modular steps, ensuring no loss of functionality.
- Users can define rules, branching, and event-driven logic for advanced workflows.
- The system is designed for extensibility: new steps, detectors, or output types can be added without breaking existing workflows.

### Dynamic Pipeline Example
- User selects input(s), configures steps and parameters, and saves the workflow.
- The pipeline is built and executed dynamically, with all outputs, metadata, and lineage tracked and logged.

### Error Handling and Validation
- Each step is validated before execution.
- Regression testing ensures all legacy workflows can be recreated and run in the new system.

## Key Patterns
- **Pipeline Pattern:** Modular, user-defined step chaining.
- **Strategy Pattern:** User selects/configures strategies for each step (e.g., model, thresholds).
- **Rule/Branching Pattern:** Users can define event-driven or conditional logic in workflows.
- **Extensibility:** New steps and features can be added as modules.

### Core Components
1. **Audio Processing Pipeline (Pyannote-First Workflow - Implemented)**
   ```mermaid
   flowchart TD
       Input[Input File/URL/Folder] --> ProcessCheck{Input Type?}
       ProcessCheck -- File/URL --> PreProcess
       ProcessCheck -- Folder --> FindNewest[Find Newest Compatible]
       FindNewest --> PreProcess
       
       subgraph PreProcess
            direction LR
            PreProcess --> VidCheck{Video?}
            VidCheck -- Yes --> Extract[ffmpeg: Extract Audio]
            VidCheck -- No ----> UseOriginal[Use Original Audio]
            Extract --> NormalizedInput
            UseOriginal --> NormalizedInput
            NormalizedInput --> Normalize[ffmpeg: Normalize Audio (-16 LUFS)]
       end

       Normalize --> OriginalNormAudio[Original Normalized Audio]

       subgraph Optional Separation
           direction LR
           OriginalNormAudio --> OptDemucs{Demucs Enabled?}
           OptDemucs -- Yes --> Demucs[Demucs: Separate Vocals]
           Demucs --> Vocals[Vocals Track]
           Demucs --> NoVocals[Non-Vocals Track]
           OptDemucs -- No ----> SkipDemucs[Use Normalized Audio Directly]
       end

       Vocals --> AudioForDiarization{Audio for Diarization}
       SkipDemucs --> AudioForDiarization

       AudioForDiarization --> Diarize[Pyannote 3.1: Diarize Speakers]
       Diarize --> DiarizationResult[Diarization Timeline]
       DiarizationResult --> MergeTurns[Merge Adjacent Turns (Same Speaker)]
       MergeTurns --> MergedBlocks[Merged Speaker Blocks]

       subgraph Optional Contextual Annotation
          direction LR
          OriginalNormAudio --> OptClap{Contextual CLAP Enabled?}
          OptClap -- Yes --> ClapEvents[CLAP: Detect Events on Full Audio]
          ClapEvents --> EventData[Contextual Event Data]
          OptClap -- No ----> SkipClap
       end

       MergedBlocks --> LoopBlocks{Loop Merged Blocks}

       subgraph Process Speaker Blocks
           direction LR
           LoopBlocks --> ExtractSegment[soundfile: Extract Original Norm. Segment]
           ExtractSegment --> SegmentAudio[Segment Audio Data]
           SegmentAudio --> Transcribe[Whisper: Transcribe Segment]
           Transcribe --> SegmentResult[Segment Result (Text, Speaker, Timestamps)]
       end

       SegmentResult --> AggregateResults[Aggregate Results]
       EventData --> AggregateResults

       AggregateResults --> BuildYAML[Build Final Dictionary]
       BuildYAML --> FinalOutput[YAML Dump (master_transcript.yaml w/ CustomDumper)]
   ```

2. **File Structure (As Implemented)**
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
           RunDir --> DiarizationDir(diarization/)
           RunDir --> EventsDir(events/)
           # Removed SoundsDir as contextual sound detection isn't implemented yet
           # RunDir --> SoundsDir(sounds/)
           RunDir --> SegmentsDir(segments/)
           RunDir --> MasterYAML(master_transcript.yaml)
           # Removed specific transcript files as main output is YAML
           # RunDir --> TranscriptsDir(transcripts/)
           # RunDir --> ResultsZip(results.zip) # Zip is optional
       end

       NormDir --> NormWav("*.wav")
       DemucsDir --> VocalsWav("*_vocals.wav")
       DemucsDir --> NoVocalsWav("*_no_vocals.wav")
       DiarizationDir --> DiarizationRTTM("diarization.rttm")
       EventsDir --> EventsJSON("contextual_events.json")
       # SoundsDir --> SoundsJSON("sounds.json")
       SegmentsDir --> SegmentWav("block_*.wav")
       # TranscriptsDir --> FullTxt("full_transcript.txt")
       # TranscriptsDir --> SegmentTxt("segment_*.txt")

       style DemucsDir fill:#f9f,stroke:#333,stroke-width:2px
       style EventsDir fill:#f9f,stroke:#333,stroke-width:2px
       # style SoundsDir fill:#f9f,stroke:#333,stroke-width:2px, BORDER-STYLE: dashed
       style SegmentsDir fill:#ccf,stroke:#333,stroke-width:2px # Segments are always created now
       # note right of DemucsDir : Dashed boxes indicate optional outputs dependent on configuration
       note right of DemucsDir : Pink boxes indicate optional outputs dependent on configuration
       note right of SegmentsDir : Blue box indicates standard intermediate output of Pyannote-first workflow
   ```
   **Subdirectory Purpose:**
   *   `normalized/`: Normalized audio derived from input.
   *   `demucs/`: Optional: Output from vocal separation (vocals, no_vocals).
   *   `diarization/`: Output from speaker diarization (RTTM format).
   *   `events/`: Optional: Detected event data from contextual CLAP run on full audio.
   *   `segments/`: Audio segments extracted based on merged diarization blocks, used for transcription.
   *   `master_transcript.yaml`: **Primary Output:** Structured YAML output containing all metadata, conversation blocks (speaker, timestamps, text, word timings if enabled), and optional contextual events.
   *   `config.yaml`: Copy of the preset configuration used for the run.
   *   `processing.log`: Log file for the run.

## Design Patterns

### 1. Pipeline Pattern
- **Pyannote-first processing:** Diarization drives segmentation.
- Sequential steps: Normalize -> [Demucs] -> Diarize -> Merge -> [CLAP] -> Extract/Transcribe Loop -> Aggregate.
- Coordinated segment handling via merged blocks.

### 2. Strategy Pattern
- Configurable behavior via `presets.py`:
    - Optional Demucs (`workflow['separate_vocals']`).
    - Optional Contextual CLAP (`workflow['detect_events_contextual']`).
    - Whisper model size (`transcription['model_size']`).
    - Word timestamps (`transcription['word_timestamps']`).
    - Diarization/Merging parameters (`diarization['merge_gap_s']`, `diarization['min_block_duration_s']`).

### 3. Helper Functions / Utility Pattern
- Dedicated functions for specific tasks:
    - `normalize_audio`
    - `separate_vocals_with_demucs`
    - `merge_diarization_turns`
    - `extract_audio_segment`
    - `run_clap_event_detection`
    - `format_speaker_label`

## Key Technical Decisions

### 1. Segmentation Strategy
- **Pyannote Speaker Diarization (3.1):** Primary method for identifying speaker turns.
- **Turn Merging Logic (`merge_diarization_turns`):** Merging adjacent turns *from the same speaker* based on `max_silence_gap` to form coherent speaker blocks.
- **Segment Extraction (`extract_audio_segment`):** Using merged block boundaries to extract corresponding audio segments from the *original normalized* audio using `soundfile` for transcription.
- **CLAP-based segmentation DEPRECATED.**

### 2. Audio Track Management
- **Original Normalized Audio:** Preserved and used as the source for segment extraction.
- **Optional Vocals Track:** Used as input for diarization if Demucs is enabled.
- **Optional Non-Vocal Track:** Saved but not actively used in the main transcription flow (potentially input for contextual CLAP).

### 3. Processing Flow
- **Speaker Block Centric:** The main processing loop iterates through the merged speaker blocks derived from diarization.
- **Transcription per Block:** Whisper is run individually on each extracted audio segment.
- **Contextual Annotation:** Optional CLAP runs on the full normalized audio and results are correlated with the final blocks.

### 4. Output Structure
- **Primary Output:** `master_transcript.yaml` containing structured data.
- **Intermediate Files:** Organized into subdirectories (`normalized`, `demucs`, `diarization`, `segments`, `events`).

## Component Relationships

### 1. Diarization & Merging
- **Input:** Normalized audio (or Vocals track if Demucs used).
- **Process:** Runs Pyannote 3.1 pipeline, applies `merge_diarization_turns` based on `max_silence_gap` and `min_block_duration`.
- **Output:** List of merged speaker blocks (dictionaries containing speaker, start, end, duration, original turns).

### 2. Segment Processing & Transcription
- **Input:** Merged speaker blocks, Original Normalized Audio path.
- **Process:** Loops through blocks:
    - Calls `extract_audio_segment` using block timestamps on the original normalized audio.
    - Runs Whisper transcription on the extracted segment WAV file.
    - Optionally correlates with pre-computed contextual CLAP events based on timestamp overlap.
- **Output:** List of dictionaries, each representing a processed block with transcription, speaker, timings, etc.

### 3. Output Generation
- **Input:** Aggregated results dictionary containing metadata, processed blocks, paths, etc.
- **Process:** Uses `yaml.dump` with a `CustomDumper` to serialize the results dictionary (handling numpy/Segment objects) into `master_transcript.yaml`.

## Error Handling
- **Try/Except Blocks:** Used around major processing steps (Normalization, Demucs, Diarization, CLAP, Transcription loop, YAML saving).
- **Status Updates:** `update_progress` helper logs step completion/errors and updates the main `results` dictionary.
- **Graceful Degradation:** Contextual CLAP failure is logged but doesn't stop the main transcription process.
- **Block-Level Errors:** Errors during segment extraction or transcription for a specific block are logged, and processing continues to the next block.
- **YAML Dumper:** Custom dumper handles potential serialization issues for specific data types.

**Overall Architecture:**
*   Core processing pipeline (`whisperBite.py::process_audio`).
*   Gradio web UI (`app.py`) - *Needs testing/potential updates for new workflow*. 
*   **Pyannote-driven Segmentation Workflow (Implemented).**

**Key Components/Functions:**
*   `app.py`:
    *   `build_interface()`: Defines UI.
    *   `run_pipeline()`: Handles UI logic, calls `whisperBite.process_audio`, passes `progress_callback`.
*   `whisperBite.py`:
    *   `process_audio()`: Main orchestrator implementing the Pyannote-first flow described above.
    *   `extract_audio_from_video()`: Uses `ffmpeg` (called from `main` or `app`).
    *   `normalize_audio()`: Uses `ffmpeg`.
    *   `merge_diarization_turns()`: Merges adjacent turns for the same speaker.
    *   `extract_audio_segment()`: Extracts segment using `soundfile`.
    *   `format_speaker_label()`: Formats speaker labels.
    *   `CustomDumper`: Handles YAML serialization for specific types.
    *   `main()`: Handles CLI execution, preset loading, input path handling, video extraction, calls `process_audio`.
*   `vocal_separation.py`:
    *   `separate_vocals_with_demucs()`: Runs Demucs via subprocess.
*   `event_detection.py`:
    *   `run_clap_event_detection()`: Used for optional contextual analysis.
*   `utils.py`:
    *   Helpers (`sanitize_filename`, `download_audio`, `zip_results`, `get_media_info`).

**Processing Flow (Implemented Pyannote-First):**
Input -> (`app.py` or `main`) -> Video Check/Extract -> `process_audio` -> Get Media Info -> Normalize -> [Demucs] -> Diarize (Pyannote) -> Merge Turns -> [Contextual CLAP] -> Loop Merged Blocks (Extract Original Segment -> Transcribe Segment w/ Whisper) -> Build Results Dict -> Write `master_transcript.yaml` (Custom Dumper) -> [Zip] -> Cleanup.

**Data Management:**
*   Unique, timestamped output directory per run.
*   Handles audio/video inputs, URLs, folders.
*   **Primary Output:** `master_transcript.yaml`.
*   Intermediate directories: `normalized/`, `demucs/` (opt), `diarization/`, `segments/`, `events/` (opt).
*   `config.yaml` and `processing.log` saved per run.

## Modular Pipeline Architecture (Ongoing)

- Demucs runs after normalization (if enabled), producing vocals and no_vocals tracks for downstream use.
- CLAP Pass 1 segments calls/conversations on normalized track using user prompts.
- For each call segment:
    - Diarization (manual speaker count if provided) on vocals track if available, else normalized.
    - VAD on vocals track if available, else normalized, to refine soundbite boundaries.
    - Whisper transcription for the full segment (with speaker labels).
    - After all annotation, cut soundbites using VAD and annotate with speaker/event data, using vocals/no_vocals/remix as per user config.
- CLAP Pass 2 runs on no_vocals for event annotation (user prompts).
- CLAP Pass 3 runs on normalized for additional annotation (user prompts).
- All user options (CLAP prompts, soundbite source, remix volumes) are respected.
- Architecture is fully modular, extensible, and user-driven. Implementation ongoing.