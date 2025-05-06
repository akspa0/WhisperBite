# Project Brief: WhisperBite

## Purpose
WhisperBite is a tool that augments audio with speaker-based transcriptions and contextual hints. It is designed to make audio content accessible, searchable, and richly detailed for both humans and downstream AI systems.

## Core Workflow
- **Demucs**: Separates vocals from the rest of the audio, providing a clean speech track.
- **pyannote**: Performs speaker diarization on the vocal track, identifying who speaks when.
- **Whisper**: Transcribes each diarized segment, producing accurate, per-speaker, time-aligned transcripts.
- **CLAP**: Annotates the normalized audio with contextual events (music, laughter, sound effects, etc.), adding rich scene context.

## Combined Output
- Generates contextual transcriptions: each speaker's words are annotated with relevant audio events, providing a scene-aware, accessible, and richly detailed record of the conversation.
- All data is structured for easy search, scene reconstruction, or further AI-driven illustration (e.g., LLMs, image generation).

## Example Use Case
- **Input**: A podcast, phone call, or comedy show recording.
- **Output**: A transcript showing who said what, when, and what was happening in the background (e.g., [S1, 00:12.3] "Hello!" [music starts], [S2, 00:14.1] "Hi!" [laughter]).

## Why?
This enables downstream applications—like LLMs or image generation tools—to reconstruct, illustrate, or summarize the scene, making audio content accessible and vivid even for those who cannot hear it.

## Project Overview
WhisperBite is an advanced audio processing tool that combines speaker diarization, transcription, and sound event detection capabilities. It processes audio/video files to produce detailed transcriptions with speaker identification and optional sound event detection.

## Core Requirements
1. Audio Processing
   - Support for various audio/video input formats
   - Automatic audio extraction from video files
   - Audio normalization and quality optimization
   - Optional stereo channel splitting

2. Speaker Diarization
   - Accurate speaker separation using pyannote.audio
   - Support for automatic speaker count detection
   - Optional second pass refinement for improved accuracy
   - Stereo channel-specific processing capability

3. Transcription
   - High-quality transcription using OpenAI's Whisper
   - Support for multiple Whisper models
   - Word-level timestamp extraction
   - Individual word audio snippet extraction

4. Sound Detection
   - CLAP-based sound event detection
   - Support for custom sound prompts
   - Vocal cue detection capability
   - Automatic audio cutting at detected events

5. Output Generation
   - Structured YAML output with metadata
   - Organized directory structure for outputs
   - Relative path handling for portability
   - Results compression for easy sharing

## Project Goals
1. Provide accurate and reliable audio transcription with speaker identification
2. Enable flexible processing workflows for different use cases
3. Maintain high performance and efficient resource usage
4. Ensure robust error handling and logging
5. Support extensibility for future enhancements 