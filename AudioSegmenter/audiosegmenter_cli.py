# audiosegmenter_cli.py
import typer
from typing import List, Optional
import os
import sys
import subprocess
import json
import re # NEW: Import re for regex operations in normalize_path
from pathlib import Path
from datetime import datetime # NEW: Import datetime for summary timestamps
from core.audio_utils import prepare_audio
from core.diarization import initialize_diarization_pipeline, run_diarization_and_segmentation
from core.clap_annotator import annotate_audio
from core.transcription import initialize_whisper_model, transcribe_segment, save_transcription_files, DEFAULT_WHISPER_MODEL

app = typer.Typer(help="AudioSegmenter: Speaker diarization, transcription, and sound annotation tool")

def check_system_dependencies():
    """Verify system dependencies (FFmpeg)"""
    try:
        # Try using subprocess to check if ffmpeg is available
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True,
                               check=False)
        
        if result.returncode == 0:
            # FFmpeg is installed and working
            return True
        else:
            raise RuntimeError("FFmpeg command failed")
            
    except (FileNotFoundError, RuntimeError):
        typer.secho(
            "Error: FFmpeg not found! Please install it:\n"
            "  Linux/WSL: sudo apt install ffmpeg\n"
            "  Windows: Download from ffmpeg.org\n"
            "  macOS: brew install ffmpeg", 
            fg=typer.colors.RED
        )
        sys.exit(1)

def normalize_path(path_str: str) -> str:
    """Normalize a path string to ensure correct separators and absolute paths.
    Handles Windows/Unix-style paths and specific concatenated date patterns."""
    
    original_path_for_debug = str(path_str) # Ensure it's a string for comparison
    current_path_str = str(path_str) # Work with a copy

    # 1. Fix missing separator for 'nameYYYY-MM-DD' pattern (inspired by run.py logic)
    #    Example: test_audio2014-06-19-aculo.mp3 -> test_audio/2014-06-19-aculo.mp3
    # This regex looks for a word-like part followed immediately by a YYYY-MM-DD date pattern.
    # It captures the part before the date, the date itself, and the rest of the string.
    # The (?<![/\\]) is a negative lookbehind to ensure there isn't already a path separator before the date part,
    # avoiding alteration of already correct paths like some/dir/prefix2024-01-01.
    # The ([a-zA-Z0-9_.-]+[a-zA-Z0-9_]) part aims to capture a reasonable prefix that doesn't end in a separator.
    match = re.match(r'([a-zA-Z0-9_.-]+[a-zA-Z0-9_])(?<![/\\])(\d{4}-\d{2}-\d{2})(.*)', current_path_str)
    if match:
        prefix = match.group(1)
        date_part = match.group(2)
        suffix = match.group(3)
        
        # Only apply if the prefix itself doesn't look like it contains the date already or is not a common extension.
        # This is a heuristic to avoid over-aggressive splitting.
        # A more robust solution might involve checking if `prefix` is a directory, but that's harder mid-parse.
        if not date_part in prefix and not prefix.lower().endswith(('.mp3', '.wav', '.mp4', '.mkv', '.flac', '.m4a')):
            corrected_path_candidate = f"{prefix}/{date_part}{suffix}"
            # Heuristic: if the prefix is short or the original path looks like a file, be cautious.
            # This is tricky. The original run.py search was broader.
            # For now, let's assume the match implies the fix is desired as per run.py's intent.
            current_path_str = corrected_path_candidate
            # print(f"Debug: Regex applied. Original: {original_path_for_debug}, Candidate by regex: {current_path_str}")

    # 2. Handle Windows backslashes (convert to forward slashes)
    if '\\' in current_path_str:
        current_path_str = current_path_str.replace('\\', '/')

    # 3. Normalize path using pathlib for robustness (handles ~/, ../, ./, absolute paths)
    try:
        resolved_path = Path(current_path_str).expanduser().resolve()
        normalized_path_str = str(resolved_path)
    except Exception as e:
        # Fallback for safety, though Path.resolve is usually robust.
        # This might happen with completely invalid path characters not handled above.
        print(f"Warning: Path.resolve failed for '{current_path_str}' (original: '{original_path_for_debug}'): {e}. Using os.path.normpath.")
        normalized_path_str = os.path.normpath(current_path_str) # os.path.normpath also handles / and \ to some extent

    # Debug printing: Print if any transformation occurred.
    if normalized_path_str != original_path_for_debug:
        if original_path_for_debug == current_path_str and match: # Path.resolve did the main work after regex maybe matched but didn't change string
             print(f"Path normalized (by Path.resolve): {original_path_for_debug} -> {normalized_path_str}")
        elif current_path_str != original_path_for_debug and match: # Regex made a change
             print(f"Fixed missing separator and normalized: {original_path_for_debug} -> {normalized_path_str}")
        else: # General normalization by Path.resolve or backslash replacement
             print(f"Path normalized: {original_path_for_debug} -> {normalized_path_str}")

    return normalized_path_str

def _seconds_to_hmsm(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"

@app.command()
def process(
    input_path: str = typer.Argument(..., help="Path to input audio/video file"),
    output_dir: str = typer.Argument(..., help="Output directory for results"),
    # Audio preprocessing
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Enable/disable audio normalization."),
    target_lufs: float = typer.Option(-14.0, help="Target LUFS level for normalization (e.g., -14.0 for YouTube)"),
    force_resample: bool = typer.Option(True, "--resample/--no-resample", help="Force/disable resample to 16kHz mono."),
    # Diarization
    hf_token: str = typer.Option(None, help="Hugging Face token (default: HF_TOKEN env var). Used for Pyannote and potentially gated HF models."),
    num_speakers: Optional[int] = typer.Option(None, help="Exact speaker count for diarization."),
    min_speakers: int = typer.Option(1, help="Minimum speaker count for diarization."),
    max_speakers: int = typer.Option(5, help="Maximum speaker count for diarization."),
    # NEW: Transcription options
    transcribe: bool = typer.Option(True, "--transcribe/--no-transcribe", help="Enable/disable speech transcription."),
    whisper_model_name: str = typer.Option(DEFAULT_WHISPER_MODEL, help=f"Whisper model name from Hugging Face. Default: {DEFAULT_WHISPER_MODEL}"),
    transcription_language: Optional[str] = typer.Option(None, help="Language code for transcription (e.g., en, es). Whisper auto-detects if None."),
    word_timestamps: bool = typer.Option(True, "--word-timestamps/--no-word-timestamps", help="Enable/disable word-level timestamps in transcription."),
    # CLAP annotation
    clap_model: str = typer.Option("microsoft/clap-htsat-unfused", help="CLAP model name for sound event annotation."),
    device: str = typer.Option("cuda", help="Inference device (e.g., cuda, cpu, mps). Auto-detects if 'cuda' selected but unavailable."),
    event_prompts: Optional[List[str]] = typer.Option(None, help="Event prompts for CLAP (comma-separated strings)."),
    event_threshold: float = typer.Option(0.5, help="Event detection threshold for CLAP."),
    sound_prompts: Optional[List[str]] = typer.Option(None, help="Sound prompts for CLAP (comma-separated strings)."), 
    sound_threshold: float = typer.Option(0.3, help="Sound detection threshold for CLAP."),
    # Debug
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output.")
):
    """Process an audio file: preprocess, diarize, segment, transcribe, and annotate sound events."""
    processing_start_time = datetime.utcnow()
    status = "success"
    error_message = None
    results_summary_outputs = {}

    try:
        # Check dependencies
        check_system_dependencies()
        
        original_input_path_for_summary = input_path
        original_output_dir_for_summary = output_dir

        # Print original paths for debugging
        if verbose: 
            print(f"Original input path: {input_path}")
            print(f"Original output directory: {output_dir}")
        
        # Normalize paths to ensure correct format
        input_path = normalize_path(input_path)
        output_dir_path = Path(normalize_path(output_dir)) # Ensure output_dir is a Path object for consistency
        output_dir = str(output_dir_path) # Keep string version for existing code expecting string
        
        if verbose:
            print(f"Normalized input path: {input_path}")
            print(f"Normalized output directory: {output_dir}")
        
        # Validate the input file exists
        if not os.path.isfile(input_path):
            typer.secho(f"Error: Input file not found: {input_path}", fg=typer.colors.RED)
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directory
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Audio preprocessing
        if verbose:
            typer.echo("⏳ Preprocessing audio...")
        prepared_audio_path_str = prepare_audio(
            input_path=input_path,
            output_dir=output_dir,
            normalize=normalize,
            target_lufs=target_lufs,
            force_resample=force_resample
        )
        results_summary_outputs["prepared_audio_file"] = Path(prepared_audio_path_str).name
        
        # 2. Speaker diarization and segmentation
        if verbose:
            typer.echo("🔊 Running speaker diarization and segmentation...")
        diarization_pipeline = initialize_diarization_pipeline(hf_token)
        diarization_results = run_diarization_and_segmentation(
            pipeline=diarization_pipeline,
            prepared_audio_path=prepared_audio_path_str,
            output_dir=output_dir,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        results_summary_outputs["rttm_file"] = diarization_results['rttm_file_path']
        results_summary_outputs["speaker_segments_manifest"] = diarization_results['speaker_segments_manifest']
        diarization_stats = {
            "identified_speakers": diarization_results['identified_speakers'],
            "total_speech_duration": diarization_results.get('total_speech_duration') # Ensure this key exists
        }
        
        if verbose:
            typer.echo(f"✅ Identified {len(diarization_results['identified_speakers'])} speakers:")
            for speaker in diarization_results['identified_speakers']:
                typer.echo(f"  - {speaker}")
            typer.echo(f"  RTTM file: {diarization_results['rttm_file_path']}")
            typer.echo(f"  Segments manifest: {diarization_results['speaker_segments_manifest']}")
        
        # 3. NEW: Speech Transcription
        full_transcript_data = []
        transcription_outputs = {}
        if transcribe:
            if verbose: typer.echo("📝 Running speech transcription...")
            
            asr_pipeline, _, whisper_device = initialize_whisper_model(model_name=whisper_model_name, device=device, hf_token=hf_token)
            if verbose: typer.echo(f"Whisper model initialized on device: {whisper_device}")

            speaker_segments_manifest_path = output_dir_path / diarization_results['speaker_segments_manifest']
            if not speaker_segments_manifest_path.exists():
                typer.secho(f"Error: Speaker segments manifest not found: {speaker_segments_manifest_path}", fg=typer.colors.RED)
            else:
                with open(speaker_segments_manifest_path, 'r', encoding='utf-8') as f:
                    speaker_segments = json.load(f)
                
                transcripts_base_dir = output_dir_path / "transcripts"
                transcripts_segments_dir = transcripts_base_dir / "speaker_segments"
                transcripts_segments_dir.mkdir(parents=True, exist_ok=True)

                # For summary: track number of transcribed segments and total transcribed duration
                num_transcribed_segments = 0
                total_transcribed_duration = 0.0

                for segment_info in speaker_segments:
                    segment_wav_path = output_dir_path / segment_info["file_path"]
                    if not segment_wav_path.exists():
                        if verbose: typer.secho(f"Warning: Segment WAV file not found: {segment_wav_path}, skipping transcription.", fg=typer.colors.YELLOW)
                        continue
                    
                    if verbose: typer.echo(f"Transcribing segment: {segment_wav_path} ({segment_info['speaker_id']} {segment_info['start_time']}-{segment_info['end_time']}) ...")
                    
                    transcription_result = transcribe_segment(
                        transcribe_pipeline=asr_pipeline,
                        audio_segment_path=str(segment_wav_path),
                        language=transcription_language,
                        return_word_timestamps=word_timestamps
                    )
                    
                    relative_segment_file_path = Path(segment_info["file_path"])
                    transcript_base_save_path = transcripts_segments_dir / relative_segment_file_path.parent.name / relative_segment_file_path.stem
                    
                    saved_files = save_transcription_files(
                        transcription_result=transcription_result,
                        base_output_path_no_ext=str(transcript_base_save_path),
                        global_start_time=segment_info["start_time"]
                    )
                    # Store relative paths for summary
                    # transcription_outputs.setdefault("individual_segment_transcripts", []).extend(
                    #    [str(Path(f).relative_to(output_dir_path)) for f in saved_files.values() if f]
                    # )

                    num_transcribed_segments += 1
                    total_transcribed_duration += (segment_info["end_time"] - segment_info["start_time"])
                    
                    text = transcription_result.get("text", "").strip()
                    word_timestamps_for_full_transcript = []

                    if "chunks" in transcription_result and transcription_result["chunks"]:
                        word_timestamps_for_full_transcript = [
                            {
                                "text": chunk["text"],
                                "timestamp": (
                                    round(chunk["timestamp"][0] + segment_info["start_time"], 3),
                                    round(chunk["timestamp"][1] + segment_info["start_time"], 3)
                                )
                            }
                            for chunk in transcription_result["chunks"]
                        ]
                    elif text:
                        word_timestamps_for_full_transcript = [{
                            "text": text,
                            "timestamp": (round(segment_info["start_time"],3), round(segment_info["end_time"],3))
                        }]

                    full_transcript_data.append({
                        "speaker_id": segment_info["speaker_id"],
                        "start_time": round(segment_info["start_time"], 3),
                        "end_time": round(segment_info["end_time"], 3),
                        "text": text,
                        "language_detected": transcription_result.get("language_detected"),
                        "word_timestamps_available": bool(word_timestamps_for_full_transcript) and \
                                                   not (len(word_timestamps_for_full_transcript) == 1 and word_timestamps_for_full_transcript[0]["text"] == text),
                        "chunks": word_timestamps_for_full_transcript
                    })
                
                if full_transcript_data:
                    full_transcript_data.sort(key=lambda x: x["start_time"])
                    full_json_path = transcripts_base_dir / "full_transcript.json"
                    with open(full_json_path, 'w', encoding='utf-8') as f:
                        json.dump(full_transcript_data, f, indent=2, ensure_ascii=False)
                    if verbose: typer.echo(f"Full transcript JSON saved to: {full_json_path}")
                    transcription_outputs["full_transcript_json"] = str(full_json_path.relative_to(output_dir_path))
                    
                    full_txt_path = transcripts_base_dir / "full_transcript.txt"
                    with open(full_txt_path, 'w', encoding='utf-8') as f:
                        for item in full_transcript_data:
                            start_hmsm = _seconds_to_hmsm(item["start_time"])
                            end_hmsm = _seconds_to_hmsm(item["end_time"])
                            f.write(f"[{start_hmsm} --> {end_hmsm}] {item['speaker_id']}: {item['text']}\n")
                    if verbose: typer.echo(f"Full transcript TXT saved to: {full_txt_path}")
                    transcription_outputs["full_transcript_txt"] = str(full_txt_path.relative_to(output_dir_path))
                    transcription_outputs["num_transcribed_segments"] = num_transcribed_segments
                    transcription_outputs["total_transcribed_duration_seconds"] = round(total_transcribed_duration, 3)
                else:
                    if verbose: typer.echo("No segments were transcribed to create a full transcript.")
        else:
            if verbose: typer.echo("Transcription disabled via --no-transcribe.")
        
        results_summary_outputs["transcription_files"] = transcription_outputs if transcribe else None

        # 4. Sound annotation (if prompts provided)
        clap_outputs = {}
        if event_prompts or sound_prompts:
            if verbose: typer.echo("🎵 Running sound annotation...")
            clap_results = annotate_audio( # Modified to get return value
                audio_path=prepared_audio_path_str,
                output_dir=output_dir,
                event_prompts=event_prompts,
                event_threshold=event_threshold,
                sound_prompts=sound_prompts,
                sound_threshold=sound_threshold,
                model_name=clap_model,
                device=device
            )
            if clap_results.get("clap_events_file"): # annotate_audio should return paths
                 clap_outputs["clap_events_file"] = str(Path(clap_results["clap_events_file"]).relative_to(output_dir_path))
            if clap_results.get("clap_sounds_file"): # annotate_audio should return paths
                 clap_outputs["clap_sounds_file"] = str(Path(clap_results["clap_sounds_file"]).relative_to(output_dir_path))
        results_summary_outputs["clap_annotation_files"] = clap_outputs if clap_outputs else None
        
    except FileNotFoundError as e:
        status = "failure"
        error_message = str(e)
        # No sys.exit here, allow finally block to run
    except Exception as e:
        status = "failure"
        error_message = f"An unexpected error occurred: {str(e)}"
        # Consider logging the full traceback here for debugging
        # import traceback
        # error_message += "\n" + traceback.format_exc()
        # No sys.exit here

    finally:
        processing_end_time = datetime.utcnow()
        processing_duration_seconds = (processing_end_time - processing_start_time).total_seconds()

        summary_data = {
            "processing_tool": "AudioSegmenter",
            "version": "0.2.0", # Update version as features are added
            "input_file_path_original": original_input_path_for_summary,
            "input_file_path_normalized": input_path if status == "success" else original_input_path_for_summary,
            "output_directory_original": original_output_dir_for_summary,
            "output_directory_normalized": output_dir if status == "success" else original_output_dir_for_summary,
            "processing_parameters": {
                "normalize_audio": normalize,
                "target_lufs": target_lufs,
                "force_resample_to_16k": force_resample,
                "hf_token_provided": bool(hf_token),
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "transcribe_enabled": transcribe,
                "whisper_model_name": whisper_model_name,
                "transcription_language": transcription_language,
                "word_timestamps_enabled": word_timestamps,
                "clap_model_name": clap_model,
                "clap_device": device, # Note: this device applies to all models in current CLI
                "clap_event_prompts": event_prompts,
                "clap_event_threshold": event_threshold,
                "clap_sound_prompts": sound_prompts,
                "clap_sound_threshold": sound_threshold,
                "verbose": verbose
            },
            "diarization_stats": diarization_stats if status == "success" and 'diarization_stats' in locals() else None,
            "outputs": results_summary_outputs,
            "processing_start_time_utc": processing_start_time.isoformat() + "Z",
            "processing_end_time_utc": processing_end_time.isoformat() + "Z",
            "processing_duration_seconds": round(processing_duration_seconds, 3),
            "status": status,
            "error_message": error_message
        }

        summary_file_path = output_dir_path / "results_summary.json"
        try:
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            if verbose and status == "success":
                typer.echo(f"📊 Results summary saved to: {summary_file_path}")
            elif status == "failure":
                 typer.secho(f"❌ Processing failed. Summary saved to: {summary_file_path}", fg=typer.colors.RED)
                 typer.secho(f"Error: {error_message}", fg=typer.colors.RED)

        except Exception as e:
            typer.secho(f"Critical Error: Could not write results_summary.json: {e}", fg=typer.colors.RED)
            if error_message: # if an error already occurred from main processing
                typer.secho(f"Original processing error was: {error_message}", fg=typer.colors.RED)
            else: # if the only error is writing the summary
                status = "failure" # ensure status reflects this failure

        if verbose and status == "success":
            typer.echo(f"✅ Processing complete! Results saved to: {output_dir}")
        
        if status == "success":
            typer.secho("Processing complete!", fg=typer.colors.GREEN)
        else:
            # No explicit exit here if we want main to return control, 
            # but for a CLI tool an exit code is common for failures.
            sys.exit(1) # Exit with error code if processing failed.

if __name__ == "__main__":
    app()