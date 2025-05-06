# Progress Status

## What Works

- ✅ All core features (input types, normalization, Demucs, Pyannote diarization, Whisper transcription, CLAP, soundbite extraction, multi-file/folder/URL input, output structure, metadata, etc.) are present and must be preserved in the refactor.
- ✅ Modularization of steps is planned so every feature can be exposed and chained in user-defined workflows.
- ✅ UI and backend will be refactored to allow users to build, save, and run custom workflows and rules.
- ✅ Regression testing and a feature checklist will ensure no loss of functionality during the transition.

## Current Status

- The project is entering a major refactor phase to support user-driven, modular workflows.
- All legacy workflows and features will be mapped to modular steps and validated in the new system.
- A comprehensive feature checklist and regression test plan are the next steps before major code changes.

## Next Steps

1. Inventory all existing features and workflows.
2. Draft a feature checklist and regression test plan.
3. Refactor backend and UI for modular, user-driven workflows.
4. Validate that all legacy workflows can be recreated and run in the new system.
5. Update documentation and memory bank to reflect the new architecture.

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