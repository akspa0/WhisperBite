# Progress Status: Modular Refactor (2025)

## What Works
- Modular pipeline architecture is defined and being implemented.
- All core features are mapped to modules with explicit input/output/settings.
- Workflow as config (`config.yaml`) is in place for reproducibility and sharing.

## Current Status
- **Migrating to a modular, multi-file codebase.**
    - Each processing step is a separate module/file.
    - Core logic, config, and utilities are separated.
    - Tests are being written for each module and workflow.
- **Implementing and validating the CLAP-driven segmentation workflow as the baseline.**
    - CLAP detects speech, ringing, and hang-up tones.
    - Workflow engine segments audio at these points.
    - Each segment is transcribed by Whisper.
- All legacy workflows are being mapped to the new system.
- Documentation and memory bank are being updated.

## Path Forward

```mermaid
flowchart TD
    Start[Start Modular Refactor] --> Scaffold[Scaffold Directory Structure]
    Scaffold --> Migrate[Migrate Modules]
    Migrate --> Test[Write Tests]
    Test --> CLAPSeg[CLAP-Driven Segmentation]
    CLAPSeg --> Validate[Validate Outputs]
    Validate --> Share[Share Workflows]
    Share --> LLM[LLM Integration (Future)]
```

- Scaffold and migrate to a modular, multi-file codebase.
- Implement and validate the CLAP-driven segmentation workflow.
- Expand the rules engine/editor for more versatile workflows after baseline validation.
- Update documentation and memory bank as needed.

## Modular Pipeline Refactor (Ongoing)

- Demucs runs after normalization (if enabled).
- CLAP Pass 1 segments calls on normalized track using user prompts.
- For each call segment: diarization (manual speaker count if provided), VAD, transcription, and soundbite extraction (vocals/no_vocals/remix as per config).
- CLAP Pass 2/3 for event annotation (user prompts, correct tracks).
- All user options (prompts, soundbite source, remix) are respected.
- Implementation ongoing. Next: fill in per-segment logic and modular soundbite extraction.

## 2025-05-04: Output Parity Regression

- Modular pipeline outputs do **not** match the @older/evenolder implementation (transcripts, soundbites, output structure).
- CLAP was meant to automate input segmentation for the old pipeline, **not** to change output logic or format.
- Current regressions: output format, transcript/soundbite content, and structure are not as expected.
- **Immediate priority:** Restore output parity with the old version, using CLAP only for coarse segmentation, and running the old pipeline logic on each segment.
- All further modularization and feature work is blocked until this is fixed.