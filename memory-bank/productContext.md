# Product Context

## Why This Project Exists

To process an input audio file and produce:
- Speaker diarization results (who spoke when) using `pyannote/speaker-diarization-3.1`.
- Speaker-separated audio segments based on the diarization.
- Timestamps and labels for detected sound events/annotations using a CLAP model.
- **NEW:** Accurate, speaker-attributed transcriptions of the audio content using Whisper `large-v3`.

This tool is designed to be a focused, modular component whose output can be consumed by other audio processing tools or pipelines, providing a comprehensive analysis of audio content.

## Problems It Solves

- Automates speaker diarization, sound event annotation, and **NEW: speech transcription,** significantly reducing manual effort.
- Provides structured outputs (RTTM, segmented audio, JSON manifests, **NEW: transcription files**) for integration into larger workflows.
- Handles audio preprocessing (normalization, resampling) to ensure compatibility with models.
- **NEW:** Enables content searchability and analysis through transcribed text.

## User Experience Goals

- **Ease of Use**: Simple CLI interface with clear configuration options for all processing stages.
- **Modularity**: Outputs can be easily consumed by other tools.
- **Performance**: Efficient processing of audio files with minimal setup, with considerations for resource-intensive models like Whisper.
- **NEW: Accuracy**: Prioritize high-quality outputs for diarization, annotation, and especially transcription. 