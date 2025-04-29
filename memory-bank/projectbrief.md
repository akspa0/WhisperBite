# WhisperBite Project Brief

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