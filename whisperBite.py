import os
import sys
import argparse
import datetime
import logging
import json
import subprocess
import traceback
import re
import yaml
from typing import Dict, List, Tuple, Optional, Any
import threading
import time

# Third-party imports
import torch
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, ClapModel, ClapProcessor

# Local imports
from utils import sanitize_filename, download_audio, zip_results, get_media_info, get_audio_channels
from vocal_separation import separate_vocals_with_demucs, enhance_vocals, export_audio_segments
from sound_detection import (
    detect_sound_events,
    detect_and_cut_audio,
    cut_audio_at_detections,
    cut_audio_between_events,
    extract_soundbites,
    TARGET_SOUND_PROMPTS,
    DEFAULT_CALL_CHUNK_DURATION,
    DEFAULT_CALL_THRESHOLD,
    CLAP_SAMPLE_RATE
)
from event_detection import run_clap_event_detection, DEFAULT_EVENTS, CLAP_SAMPLE_RATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv')

def format_speaker_label(label):
    """Formats pyannote speaker labels (e.g., SPEAKER_00, 1) to S0, S1 etc."""
    if isinstance(label, int):
        return f"S{label}"
    elif isinstance(label, str):
        # Extract number, handling potential "SPEAKER_" prefix
        parts = label.split('_')
        num_part = parts[-1]
        try:
            num = int(num_part)
            return f"S{num}"
        except ValueError:
            logging.warning(f"Could not format speaker label '{label}', using original.")
            return label # Fallback to original label if parsing fails
    else:
        logging.warning(f"Unexpected speaker label type: {type(label)}. Using as is.")
        return str(label) # Fallback for unexpected types

def normalize_audio(input_audio, output_dir, target_lufs=-16):
    """Normalize audio to target LUFS using ffmpeg."""
    normalized_dir = os.path.join(output_dir, "normalized")
    os.makedirs(normalized_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    output_file = os.path.join(normalized_dir, f"{base_name}_normalized.wav")
    
    # Use ffmpeg for consistent normalization
    cmd = [
        "ffmpeg", "-y", "-i", input_audio,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", "48000", "-acodec", "pcm_s16le", output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logging.info(f"Normalized audio saved to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logging.error(f"Error normalizing audio: {e}")
        logging.error(f"STDERR: {e.stderr.decode('utf-8')}")
        raise

def detect_optimal_speakers(diarization_pipeline, audio_file, min_speakers=1, max_speakers=10):
    """Find optimal speaker count using clustering quality metrics."""
    best_score = -float('inf')
    best_num_speakers = 2  # Default fallback
    
    # Try different speaker counts and use silhouette score to evaluate
    for num_speakers in range(min_speakers, min(max_speakers + 1, 6)):  # Limit to reasonable range
        try:
            diarization = diarization_pipeline(audio_file, num_speakers=num_speakers)
            
            # Extract speaker segments for clustering quality assessment
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.end - turn.start,
                    'speaker': speaker
                })
            
            # Simplistic quality metric: longer segments are better (less fragmentation)
            avg_segment_duration = sum(s['duration'] for s in segments) / len(segments) if segments else 0
            num_speaker_changes = len(segments) - 1
            
            # Penalize excessive fragmentation while promoting longer segments
            score = avg_segment_duration * 10 - num_speaker_changes * 0.1
            
            logging.info(f"Speaker count {num_speakers}: score={score:.2f} (avg_duration={avg_segment_duration:.2f}s, changes={num_speaker_changes})")
            
            if score > best_score:
                best_score = score
                best_num_speakers = num_speakers
        
        except Exception as e:
            logging.warning(f"Error evaluating {num_speakers} speakers: {e}")
            continue
    
    logging.info(f"Selected optimal speaker count: {best_num_speakers}")
    return best_num_speakers

def slice_audio_by_speaker(file_path, diarization, speaker_output_dir, min_segment_duration=1.0, speaker_suffix="", force_mono_output=False):
    """Slice audio by speakers based on diarization results."""
    audio = AudioSegment.from_file(file_path)
    os.makedirs(speaker_output_dir, exist_ok=True)

    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start
        
        # Skip very short segments
        if duration < min_segment_duration:
            continue
            
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
            
        speaker_segments[speaker].append({
            'start': turn.start,
            'end': turn.end,
            'duration': duration
        })

    # Create speaker directories and slice audio
    speaker_files = {}
    segment_info = {}
    segment_counter = 0
    
    for speaker_raw, segments in speaker_segments.items():
        formatted_speaker = format_speaker_label(speaker_raw) + speaker_suffix
        speaker_dir = os.path.join(speaker_output_dir, formatted_speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        
        if formatted_speaker not in speaker_files:
            speaker_files[formatted_speaker] = []
            segment_info[formatted_speaker] = []
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        
        # Merge very close segments from the same speaker (fix over-segmentation)
        merged_segments = []
        if segments:
            current = segments[0]
            for next_segment in segments[1:]:
                gap = next_segment['start'] - current['end']
                
                # If gap is small (less than 0.5s), merge segments
                if gap < 0.5:
                    current['end'] = next_segment['end']
                    current['duration'] = current['end'] - current['start']
                else:
                    merged_segments.append(current)
                    current = next_segment
                    
            merged_segments.append(current)
        
        # Extract and save the merged segments
        for i, segment in enumerate(merged_segments):
            segment_audio = audio[segment['start'] * 1000:segment['end'] * 1000]
            
            # Apply a slight fade in/out to avoid clicks
            fade_duration = min(100, segment_audio.duration_seconds * 1000 / 4)  # 100ms or 1/4 of segment
            segment_audio = segment_audio.fade_in(int(fade_duration)).fade_out(int(fade_duration))
            
            # Format: Speaker_X/0000_segment_START_END_DURATION.wav
            segment_filename = f"{segment_counter:04d}_segment_{int(segment['start'])}_{int(segment['end'])}_{int(segment['duration'])}.wav"
            segment_path = os.path.join(speaker_dir, segment_filename)
            
            # <<< Force mono if requested >>>
            if force_mono_output:
                segment_audio = segment_audio.set_channels(1)
                
            # Export with normalized volume
            segment_audio.export(segment_path, format="wav")
            
            speaker_files[formatted_speaker].append(segment_path)
            segment_info[formatted_speaker].append({
                'path': segment_path,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'sequence': segment_counter
            })
            
            segment_counter += 1
            
    # Create a JSON file with all segment information (useful for debugging/analytics)
    with open(os.path.join(speaker_output_dir, f"segments{speaker_suffix}.json"), 'w') as f:
        json.dump(segment_info, f, indent=2)
            
    # Return the dictionary containing detailed segment info, not just paths
    return segment_info

def transcribe_with_whisper(model, segment_info_dict, output_dir, enable_word_extraction=False, speaker_suffix="", force_mono_output=False):
    """Transcribe audio files using Whisper and create per-speaker transcripts."""
    all_transcriptions = {}
    word_timings = {}  # Initialize even if not used, for consistency
    word_counter = 0
    
    # Iterate through the segment info dictionary {formatted_speaker: [list_of_segment_dicts]}
    for formatted_speaker, segments in segment_info_dict.items(): # Already formatted from previous step
        speaker_transcription_dir = os.path.join(output_dir, f"{formatted_speaker}_transcriptions") # Use formatted name
        os.makedirs(speaker_transcription_dir, exist_ok=True)
        
        # Create words directory for this speaker only if needed
        words_dir = None
        if enable_word_extraction:
            words_dir = os.path.join(output_dir, f"{formatted_speaker}_words") # Use formatted name
            os.makedirs(words_dir, exist_ok=True)
        
        speaker_full_transcript = ""
        segment_transcriptions = []
        
        # Sort segments by sequence number (already dictionaries)
        segments.sort(key=lambda x: x['sequence'])
        
        # Iterate through the list of segment dictionaries for this speaker
        for segment_info in segments:
            segment_path = segment_info['path']
            start_time = segment_info['start']
            end_time = segment_info['end']
            duration = segment_info['duration']
            seq_num = segment_info['sequence']
            segment_name = os.path.basename(segment_path) # Keep for logging

            try:
                # Skip very short segments that likely won't transcribe well
                if duration < 1:
                    continue
                
                # Transcribe the segment
                # Only request word timestamps if extraction is enabled
                transcription = model.transcribe(segment_path, word_timestamps=enable_word_extraction)
                text = transcription['text'].strip()
                
                if not text:
                    continue
                
                # Extract word timings only if enabled
                if enable_word_extraction and 'segments' in transcription and transcription['segments']:
                    for segment in transcription['segments']:
                        if 'words' in segment:
                            try: # Add try-except for audio loading robustness
                                audio = AudioSegment.from_file(segment_path)
                            except Exception as audio_load_err:
                                logging.warning(f"Could not load audio file {segment_path} for word extraction: {audio_load_err}")
                                continue # Skip word extraction for this segment if audio fails

                            for word_data in segment['words']:
                                word = word_data['word'].strip()
                                if not word:
                                    continue
                                    
                                # Get word timing
                                word_start = word_data['start']
                                word_end = word_data['end']
                                word_duration = word_end - word_start
                                
                                # Skip very short words
                                if word_duration < 0.1:
                                    continue
                                    
                                # Extract word audio with generous padding
                                # Add substantial padding before and after the word (100ms or 40% of duration, whichever is greater)
                                padding_ms = max(100, int(word_duration * 1000 * 0.4))
                                
                                # Calculate padded boundaries (ensuring we don't go out of bounds)
                                word_start_ms = max(0, int((word_start * 1000) - padding_ms))
                                word_end_ms = min(len(audio), int((word_end * 1000) + padding_ms))
                                
                                # Extract with padding
                                word_audio = audio[word_start_ms:word_end_ms]
                                
                                # Apply very gentle fade in/out (10% of duration)
                                fade_ms = max(30, int(word_audio.duration_seconds * 1000 * 0.1))
                                word_audio = word_audio.fade_in(fade_ms).fade_out(fade_ms)
                                
                                # <<< Force mono if requested >>>
                                if force_mono_output:
                                    word_audio = word_audio.set_channels(1)
                                    
                                # Save word audio
                                word_filename = f"{word_counter:04d}_{word.replace(' ', '_')}.wav"
                                word_path = os.path.join(words_dir, word_filename)
                                word_audio.export(word_path, format="wav")
                                
                                # Store word timing info
                                if formatted_speaker not in word_timings:
                                    word_timings[formatted_speaker] = []
                                    
                                word_timings[formatted_speaker].append({
                                    'word': word,
                                    'file': word_path,
                                    'start': start_time + word_start,
                                    'end': start_time + word_end,
                                    'duration': word_duration,
                                    'sequence': word_counter
                                })
                                
                                word_counter += 1
                
                # Create a reasonable output filename based on the first few words and sequence
                first_words = text.split()[:5]
                # NOTE: Filename doesn't include speaker label, just sequence and content
                base_name = f"{seq_num:04d}_{sanitize_filename('_'.join(first_words))}"
                
                # Create output paths
                transcription_file = os.path.join(speaker_transcription_dir, f"{base_name}.txt")
                audio_output_file = os.path.join(speaker_transcription_dir, f"{base_name}.wav")
                
                # Write transcription to file
                # Use exact start/end times from segment_info
                with open(transcription_file, "w", encoding='utf-8') as f: 
                    timestamp_str = f"[{start_time:.2f}s - {end_time:.2f}s]"
                    f.write(f"{timestamp_str} {text}")
                
                # Copy audio file with the new name
                import shutil
                shutil.copy2(segment_path, audio_output_file)
                
                # Add to the full transcript for this speaker
                speaker_full_transcript += f"{timestamp_str} {text}\n\n"
                
                # Store info for master transcript
                segment_transcriptions.append({
                    'speaker': formatted_speaker,
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'audio_file': audio_output_file,
                    'transcript_file': transcription_file,
                    'sequence': seq_num
                })
                
            except Exception as e:
                logging.error(f"Error transcribing segment {segment_path}: {e}")
                continue
        
        # Write the combined transcript for this speaker
        if speaker_full_transcript:
            # Use formatted name for the file
            with open(os.path.join(output_dir, f"{formatted_speaker}_full_transcript.txt"), 'w') as f:
                f.write(f"=== SPEAKER {formatted_speaker} TRANSCRIPT ===\n\n") # Use formatted name in header
                f.write(speaker_full_transcript)
        
        all_transcriptions[formatted_speaker] = segment_transcriptions # Store with formatted key
    
    # Save word timings to JSON (only if enabled and populated)
    if enable_word_extraction and word_timings:
        with open(os.path.join(output_dir, "word_timings.json"), 'w') as f:
            json.dump(word_timings, f, indent=2)
    elif enable_word_extraction:
        logging.info("Word extraction enabled, but no words were extracted.")
    
    # Create a master transcript with all speakers in chronological order
    all_segments = []
    for speaker, segments in all_transcriptions.items():
        all_segments.extend(segments)
    
    # Sort by start time (more reliable than sequence if passes differ)
    all_segments.sort(key=lambda x: x['start'])
    
    # --- Master Transcript writing logic moved to process_audio --- 
    # Return all segments for merging/writing later
    return all_segments 

def run_second_pass_diarization(first_pass_segment_info, first_pass_output_dir, diarization_pipeline, whisper_model, final_output_dir, segment_min_duration=5.0, second_pass_speakers=2, force_mono_output=False):
    """
    Performs a second pass of diarization and transcription on segments 
    from the first pass to refine speaker separation.
    """
    logging.info("Starting Second Pass Diarization Refinement...")
    second_pass_base_dir = os.path.join(final_output_dir, "2nd_pass")
    second_pass_speakers_dir = os.path.join(second_pass_base_dir, "speakers")
    second_pass_transcripts_dir = os.path.join(second_pass_base_dir, "transcriptions")
    
    os.makedirs(second_pass_speakers_dir, exist_ok=True)
    os.makedirs(second_pass_transcripts_dir, exist_ok=True)
    
    refined_segments_info = [] # List to store info for the master transcript
    sub_segment_counter = 0 # Global counter for unique sub-segment filenames

    # Iterate through speakers and their first-pass segments (already dicts with formatted labels)
    for formatted_speaker, segments in first_pass_segment_info.items():
        logging.info(f"Processing {formatted_speaker} segments for second pass...")
        
        # Ensure speaker-specific directories exist in 2nd pass output (use formatted label)
        current_spk_2nd_pass_audio_dir = os.path.join(second_pass_speakers_dir, formatted_speaker)
        current_spk_2nd_pass_ts_dir = os.path.join(second_pass_transcripts_dir, f"{formatted_speaker}_transcriptions")
        os.makedirs(current_spk_2nd_pass_audio_dir, exist_ok=True)
        os.makedirs(current_spk_2nd_pass_ts_dir, exist_ok=True)

        for segment_info in segments: # segments is now expected to be the list of dicts from slice_audio_by_speaker
            segment_path = segment_info['path']
            original_start_time = segment_info['start']
            end_time = segment_info['end']
            original_duration = segment_info['duration']
            original_sequence = segment_info['sequence']

            # Skip segments shorter than the minimum duration for refinement
            if original_duration < segment_min_duration:
                logging.debug(f"Skipping short segment {os.path.basename(segment_path)} ({original_duration:.2f}s)")
                continue

            logging.info(f"Analyzing segment: {os.path.basename(segment_path)} (Duration: {original_duration:.2f}s)")

            try:
                # Run diarization on this specific segment
                # Use the default or passed-in number of speakers for the second pass
                segment_diarization = diarization_pipeline(segment_path, num_speakers=second_pass_speakers)
                
                # Check if more than one speaker was significantly detected
                speaker_turns = {}
                for turn, _, spk_label in segment_diarization.itertracks(yield_label=True):
                    if spk_label not in speaker_turns:
                        speaker_turns[spk_label] = 0
                    speaker_turns[spk_label] += (turn.end - turn.start)
                
                significant_speakers = [spk for spk, dur in speaker_turns.items() if dur > 0.5] # Consider speakers with > 0.5s total duration significant

                if len(significant_speakers) <= 1:
                    logging.info(f"  -> No significant speaker overlap detected. Skipping refinement.")
                    continue # No refinement needed for this segment

                logging.info(f"  -> Refinement needed: Detected {len(significant_speakers)} speakers. Re-slicing and transcribing...")

                # Load the original segment audio
                segment_audio = AudioSegment.from_file(segment_path)

                # Re-slice based on the *new* diarization for this segment
                for turn, _, sub_speaker in segment_diarization.itertracks(yield_label=True):
                    sub_start = turn.start
                    sub_end = turn.end
                    sub_duration = sub_end - sub_start

                    # Skip very short sub-segments resulting from re-slicing
                    if sub_duration < 0.5: 
                        continue

                    # Extract sub-segment audio
                    # Timestamps are relative to the start of the *current* segment
                    sub_segment_audio = segment_audio[sub_start * 1000 : sub_end * 1000]

                    # Apply fade
                    fade_duration = min(50, sub_segment_audio.duration_seconds * 1000 / 4) 
                    sub_segment_audio = sub_segment_audio.fade_in(int(fade_duration)).fade_out(int(fade_duration))

                    # --- Create unique filename for the sub-segment ---
                    # Use absolute start time for sorting
                    abs_start = original_start_time + sub_start
                    abs_end = original_start_time + sub_end
                    
                    # Export the temporary refined audio sub-segment to transcribe it
                    # We need a temporary path because the final name depends on the transcription
                    temp_sub_segment_path_wav = os.path.join(current_spk_2nd_pass_audio_dir, f"temp_subsegment_{sub_segment_counter}.wav")
                    sub_segment_audio.export(temp_sub_segment_path_wav, format="wav")

                    # Transcribe the sub-segment (no word timestamps needed for 2nd pass)
                    try:
                        transcription = whisper_model.transcribe(temp_sub_segment_path_wav, word_timestamps=False)
                        text = transcription['text'].strip()
                        
                        # Remove temporary file after transcription
                        try:
                            os.remove(temp_sub_segment_path_wav)
                        except OSError as e:
                            logging.warning(f"Could not remove temporary sub-segment file {temp_sub_segment_path_wav}: {e}")

                        if text:
                            # --- Generate final filename based on content --- 
                            first_words = text.split()[:5]
                            # Add sub_segment_counter for uniqueness if first words are identical
                            base_name = f"{original_sequence:04d}_{sub_segment_counter:04d}_{sanitize_filename('_'.join(first_words))}"
                            
                            final_sub_segment_path_wav = os.path.join(current_spk_2nd_pass_audio_dir, f"{base_name}.wav")
                            final_sub_segment_path_txt = os.path.join(current_spk_2nd_pass_ts_dir, f"{base_name}.txt")
                            
                            # <<< Force mono if requested >>>
                            if force_mono_output:
                                sub_segment_audio = sub_segment_audio.set_channels(1)
                                
                            # Re-exporting might be safer if temp removal failed? Let's re-export.
                            sub_segment_audio.export(final_sub_segment_path_wav, format="wav")
                            # --- End filename generation --- 

                            # Write transcription file with the new name
                            with open(final_sub_segment_path_txt, "w", encoding='utf-8') as f:
                                timestamp_str = f"[{abs_start:.2f}s - {abs_end:.2f}s]" # Use more precision
                                f.write(f"{timestamp_str} {text}")

                            # Store info for the final 2nd pass master transcript (using new paths)
                            # Use formatted speaker label from the *second pass* diarization
                            sub_speaker_formatted = format_speaker_label(sub_speaker)
                            refined_segments_info.append({
                                'speaker': sub_speaker_formatted, # The FORMATTED speaker label FROM THE SECOND PASS
                                'start': abs_start,
                                'end': abs_end,
                                'text': text,
                                'audio_file': final_sub_segment_path_wav, # Use new path
                                'transcript_file': final_sub_segment_path_txt, # Use new path
                                'sequence': sub_segment_counter # Still useful for internal tracking if needed
                            })
                        else:
                             logging.warning(f"  -> Sub-segment from {segment_path} at {abs_start:.2f}s resulted in empty transcription.")

                    except Exception as sub_transcribe_err:
                        logging.error(f"  -> Error transcribing sub-segment generated from {segment_path} at {abs_start:.2f}s: {sub_transcribe_err}")
                        # Clean up temp file even if transcription failed
                        if os.path.exists(temp_sub_segment_path_wav):
                            try:
                                os.remove(temp_sub_segment_path_wav)
                            except OSError as e:
                                logging.warning(f"Could not remove temporary sub-segment file {temp_sub_segment_path_wav} after error: {e}")

                    sub_segment_counter += 1

            except Exception as segment_process_err:
                logging.error(f"Error processing segment {segment_path} for second pass: {segment_process_err}")
                import traceback
                logging.error(traceback.format_exc())
                continue # Move to the next segment

    # Create the 2nd pass master transcript
    refined_segments_info.sort(key=lambda x: x['start']) # Sort by absolute start time

    logging.info("Second Pass Diarization Refinement Finished.")
    return refined_segments_info # Return the list

def extract_audio_from_video(video_path, output_wav_path):
    """Extracts audio from video file using ffmpeg."""
    logging.info(f"Extracting audio from video: {video_path}")
    try:
        # Use ffmpeg to extract audio as 16-bit PCM WAV, 48kHz, mono
        # -vn: disable video recording
        # -acodec pcm_s16le: standard WAV audio codec
        # -ar 48000: audio sample rate
        # -ac 1: mono audio channel
        # -y: overwrite output file if exists
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
            output_wav_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Successfully extracted audio to: {output_wav_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio using ffmpeg: {e}")
        logging.error(f"ffmpeg command: {' '.join(e.cmd)}")
        logging.error(f"ffmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logging.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during audio extraction: {e}")
        return False

def make_path_relative(absolute_path, base_dir):
    """Convert absolute paths to relative paths for YAML output."""
    if not absolute_path:
        return None
    try:
        return os.path.relpath(absolute_path, base_dir)
    except ValueError:
        logging.warning(f"Could not make path relative: {absolute_path}")
        return absolute_path

def get_audio_from_input(input_path: str, output_dir: str) -> str:
    """
    Get audio from input file, handling video files if necessary.
    Returns path to audio file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # Check if input is already an audio file
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac'}
    if os.path.splitext(input_path)[1].lower() in audio_extensions:
        return input_path
        
    # Extract audio from video if needed
    audio_output = os.path.join(output_dir, "extracted_audio.wav")
    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Use WAV codec
            '-ar', '48000',  # Set sample rate to 48kHz
            '-ac', '2',  # Set to stereo
            audio_output
        ], check=True)
        return audio_output
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract audio: {e}")
        raise

def process_stereo_channels(
    audio_path: str,
    output_dir: str,
    pipeline: Pipeline,
    model: Any,
    num_speakers: int,
    auto_speakers: bool,
    enable_word_extraction: bool,
    force_mono_output: bool
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process left and right channels of stereo audio separately.
    Returns tuple of (left_channel_segments, right_channel_segments).
    """
    # Split stereo into separate channels
    audio = AudioSegment.from_file(audio_path)
    
    # Extract left channel
    left_channel = audio.split_to_mono()[0]
    left_path = os.path.join(output_dir, "left_channel.wav")
    left_channel.export(left_path, format="wav")
    
    # Extract right channel
    right_channel = audio.split_to_mono()[1]
    right_path = os.path.join(output_dir, "right_channel.wav")
    right_channel.export(right_path, format="wav")
    
    # Process each channel
    try:
        # Process left channel
        left_speaker_dir = os.path.join(output_dir, "speakers_left")
        os.makedirs(left_speaker_dir, exist_ok=True)
        
        actual_speakers_left = num_speakers
        if auto_speakers:
            try:
                actual_speakers_left = detect_optimal_speakers(pipeline, left_path)
            except Exception as e:
                logging.warning(f"Auto speaker detection failed for left channel: {e}")
                actual_speakers_left = num_speakers
                
        diarization_left = pipeline(left_path, num_speakers=actual_speakers_left)
        segments_left = slice_audio_by_speaker(
            left_path, 
            diarization_left, 
            left_speaker_dir,
            force_mono_output=force_mono_output
        )
        processed_left = transcribe_with_whisper(
            model,
            segments_left,
            output_dir,
            enable_word_extraction=enable_word_extraction,
            force_mono_output=force_mono_output
        )
        
        # Process right channel
        right_speaker_dir = os.path.join(output_dir, "speakers_right")
        os.makedirs(right_speaker_dir, exist_ok=True)
        
        actual_speakers_right = num_speakers
        if auto_speakers:
            try:
                actual_speakers_right = detect_optimal_speakers(pipeline, right_path)
            except Exception as e:
                logging.warning(f"Auto speaker detection failed for right channel: {e}")
                actual_speakers_right = num_speakers
                
        diarization_right = pipeline(right_path, num_speakers=actual_speakers_right)
        segments_right = slice_audio_by_speaker(
            right_path, 
            diarization_right, 
            right_speaker_dir,
            force_mono_output=force_mono_output
        )
        processed_right = transcribe_with_whisper(
            model,
            segments_right,
            output_dir,
            enable_word_extraction=enable_word_extraction,
            force_mono_output=force_mono_output
        )
        
        # Add channel information to segments
        for segment in processed_left:
            segment['channel'] = 'left'
        for segment in processed_right:
            segment['channel'] = 'right'
            
        return processed_left, processed_right
        
    except Exception as e:
        logging.error(f"Error processing stereo channels: {e}")
        raise
    finally:
        # Cleanup temporary channel files
        for path in [left_path, right_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {path}: {e}")

def cut_audio_at_boundaries(audio_path: str, detected_events: list, output_dir: str) -> list:
    """
    Cut audio file at detected sound event boundaries.
    
    Args:
        audio_path (str): Path to the audio file to cut
        detected_events (list): List of detected sound events with timestamps
        output_dir (str): Directory to save cut segments
        
    Returns:
        list: List of dictionaries containing segment information
    """
    return cut_audio_at_detections(audio_path, detected_events, output_dir)

def process_audio(
    input_file: str,
    output_dir: str,
    preset_name: str,
    preset_config: Dict[str, Any],
    stop_event: Optional[threading.Event] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Process audio file according to preset configuration. Handles two-pass workflow.
    """
    if stop_event and stop_event.is_set():
        return {"status": "cancelled"}
        
    workflow = preset_config.get("workflow", {})
    
    # Use the provided device string to create a torch device object
    try:
        processing_device = torch.device(device)
        logging.info(f"Using processing device: {processing_device}")
    except Exception as e:
        logging.warning(f"Invalid device specified '{device}', falling back to auto-detect. Error: {e}")
        processing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using fallback processing device: {processing_device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary with new structure
    results = {
        "input_file": input_file,
        "output_dir": output_dir,
        "preset": preset_name,
        "audio_path": None, # Will be set after normalization/resampling
        "processing_time": datetime.datetime.now().isoformat(),
        "device": device,
        "pass1_events": {}, # Store Pass 1 results here
        "segments": [] # Store results for each processed segment
    }
    
    # <<< Centralized CLAP Model/Processor Loading (FOR PASS 1) >>>
    clap_model_pass1 = None
    clap_processor_pass1 = None
    if workflow.get("detect_events"): # Check if Pass 1 detection is needed
        logging.info("Workflow requires Pass 1 event detection, attempting to load CLAP model...")
        try:
            t_clap_load_start = time.time()
            clap_model_pass1 = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(processing_device)
            clap_processor_pass1 = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            clap_model_pass1.eval() 
            logging.info(f"CLAP model/processor (Pass 1) loaded successfully in {time.time() - t_clap_load_start:.2f}s")
        except Exception as e:
            logging.error(f"Failed to load CLAP model/processor for Pass 1 event detection: {e}", exc_info=True)
            # Continue, subsequent steps will check if model loaded
    
    # Extract and normalize audio
    normalized_audio_path = normalize_audio(input_file, output_dir)
    logging.info(f"Normalized audio path: {normalized_audio_path}")
    
    # --- Ensure Audio is 48kHz using FFmpeg --- 
    audio_path_for_processing = normalized_audio_path 
    try:
        info = sf.info(normalized_audio_path)
        current_sr = info.samplerate
        logging.info(f"Current sample rate after normalization: {current_sr}Hz")
        
        if current_sr != 48000:
            logging.warning(f"Sample rate is {current_sr}Hz, resampling to 48000Hz using ffmpeg.")
            resampled_path = os.path.join(os.path.dirname(normalized_audio_path), 
                                        f"{os.path.splitext(os.path.basename(normalized_audio_path))[0]}_48khz.wav")
            cmd = ["ffmpeg", "-y", "-i", normalized_audio_path, "-ar", "48000", resampled_path]
            t_start_ffmpeg = time.time()
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)
            t_end_ffmpeg = time.time()
            if process.returncode == 0:
                logging.info(f"FFmpeg resampling successful in {t_end_ffmpeg - t_start_ffmpeg:.2f}s. Output: {resampled_path}")
                audio_path_for_processing = resampled_path 
            else:
                logging.error(f"FFmpeg resampling failed! Return Code: {process.returncode}")
                logging.error(f"FFmpeg stderr: {process.stderr}")
                logging.error("Proceeding with non-resampled audio. Detection might fail.")
        else:
            logging.info("Audio is already 48kHz. No ffmpeg resampling needed.")
    except Exception as e:
        logging.error(f"Error checking/resampling audio sample rate: {e}", exc_info=True)
        logging.error("Proceeding with potentially incorrect sample rate audio.")
    # --- End 48kHz Check --- 
    
    logging.info(f"Final audio path for detection/processing: {audio_path_for_processing}")
    results["audio_path"] = os.path.relpath(audio_path_for_processing, output_dir)
    
    # --- Load Full Audio Data Once --- (Moved slightly earlier)
    audio_data_np = None
    audio_sr = None
    if workflow.get("detect_events") or workflow.get("cut_between_events") or workflow.get("annotate_segments") or workflow.get("transcribe"):
        try:
            logging.info(f"Loading audio file into memory: {audio_path_for_processing}")
            t_audio_load_start = time.time()
            audio_data_np, audio_sr = sf.read(audio_path_for_processing, dtype='float32')
            logging.info(f"Audio loaded into memory in {time.time() - t_audio_load_start:.2f}s (Sample Rate: {audio_sr}Hz)")
            if audio_data_np.ndim > 1: # Ensure mono
                logging.warning(f"Flattening audio to mono. Shape: {audio_data_np.shape}")
                audio_data_np = audio_data_np[:, 0]
            if audio_sr != CLAP_SAMPLE_RATE:
                logging.error(f"Loaded audio SR ({audio_sr}) != CLAP SR ({CLAP_SAMPLE_RATE}). Detection may fail.")
        except Exception as e:
            logging.error(f"Failed to load audio file {audio_path_for_processing} into memory: {e}", exc_info=True)
            audio_data_np = None

    # <<< PASS 1: Event Detection >>>
    pass1_event_results = {}
    if workflow.get("detect_events") and clap_model_pass1 and clap_processor_pass1 and audio_data_np is not None:
        logging.info("--- Running Pass 1 CLAP Event Detection ---")
        event_config = preset_config.get("event_detection", {}) 
        target_events = event_config.get("target_events", ["ringing phone", "hang-up tones"]) 
        threshold = event_config.get("threshold", 0.5)
        chunk_duration = event_config.get("chunk_duration_s", 5.0)
        min_gap = event_config.get("min_duration", 1.0) 
        if not target_events: target_events = ["ringing phone", "hang-up tones"]
        logging.info(f"Pass 1 Config: Threshold={threshold}, Chunk={chunk_duration}s, Min Gap={min_gap}s")
        logging.info(f"Pass 1 Target Events: {target_events}")
        try:
            pass1_event_results = run_clap_event_detection(
                audio_data=audio_data_np, sample_rate=audio_sr,
                clap_model=clap_model_pass1, clap_processor=clap_processor_pass1,
                device=processing_device, target_events=target_events,
                threshold=threshold, chunk_duration=chunk_duration, min_gap=min_gap
            )
            results["pass1_events"] = pass1_event_results 
            events_output_dir = os.path.join(output_dir, "events")
            os.makedirs(events_output_dir, exist_ok=True)
            events_output_path = os.path.join(events_output_dir, "pass1_events.json")
            try:
                with open(events_output_path, 'w') as f: json.dump(pass1_event_results, f, indent=2)
                logging.info(f"Saved Pass 1 event detection results to {events_output_path}")
                results["pass1_detected_events_path"] = os.path.relpath(events_output_path, output_dir)
            except Exception as e: logging.error(f"Failed to save Pass 1 event detection results: {e}")
        except Exception as e:
            logging.error(f"Error during Pass 1 run_clap_event_detection: {e}", exc_info=True)
            results["pass1_events"] = {}
            pass1_event_results = {}
    elif workflow.get("detect_events"):
        logging.warning("Skipping Pass 1 event detection: Model/audio load failure.")
    
    # Cleanup Pass 1 CLAP model
    if clap_model_pass1:
        del clap_model_pass1, clap_processor_pass1
        if processing_device.type == 'cuda': torch.cuda.empty_cache()
        logging.info("Pass 1 CLAP model/processor released.")

    # <<< Cut Audio Between Events >>>
    conversation_segments = []
    if workflow.get("cut_between_events"):
        if pass1_event_results: 
            logging.info("--- Cutting audio into conversation segments ---")
            cut_segments_output_dir = os.path.join(output_dir, "conversation_segments")
            os.makedirs(cut_segments_output_dir, exist_ok=True)
            event_config = preset_config.get("event_detection", {})
            start_types = tuple(t for t in event_config.get("target_events", []) if "ring" in t or "start" in t)
            end_types = tuple(t for t in event_config.get("target_events", []) if "hang-up" in t or "end" in t)
            if not start_types: start_types = ("ringing phone",)
            if not end_types: end_types = ("hang-up tones",)
            conversation_segments = cut_audio_between_events(
                audio_path=audio_path_for_processing, 
                all_events=pass1_event_results,
                output_dir=cut_segments_output_dir,
                start_types=start_types,
                end_types=end_types
            )
            results["conversation_segments_info"] = [
                {**seg, 'path': os.path.relpath(seg['path'], output_dir)}
                for seg in conversation_segments if seg.get('path')
            ]
        else:
            logging.warning("Skipping cutting: Pass 1 failed or no events found.")
            # If no segments, loop below won't run
    
    # >>>>>>>> START NEW SEGMENT PROCESSING LOOP HERE <<<<<<<<
    
    # --- Load models needed for segment processing (if workflow requires it) ---
    clap_model_pass2 = None
    clap_processor_pass2 = None
    whisper_model_pass2 = None

    if workflow.get("cut_between_events") and conversation_segments: # Only load if we actually have segments
        if workflow.get("annotate_segments"):
            logging.info("Workflow requires Pass 2 segment annotation, loading CLAP model...")
            try:
                t_clap_load_start = time.time()
                clap_model_pass2 = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(processing_device)
                clap_processor_pass2 = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
                clap_model_pass2.eval() 
                logging.info(f"CLAP model/processor (Pass 2) loaded successfully in {time.time() - t_clap_load_start:.2f}s")
            except Exception as e:
                logging.error(f"Failed to load CLAP model/processor for Pass 2: {e}", exc_info=True)
        
        if workflow.get("transcribe"):
            logging.info("Workflow requires segment transcription, loading Whisper model...")
            try:
                transcribe_config = preset_config.get("transcription", {})
                model_size = transcribe_config.get("model_size", "medium")
                t_whisper_load_start = time.time()
                whisper_model_pass2 = whisper.load_model(model_size, device=processing_device)
                logging.info(f"Whisper model ({model_size}) loaded for segments in {time.time() - t_whisper_load_start:.2f}s")
            except Exception as e:
                logging.error(f"Failed to load Whisper model for segment transcription: {e}", exc_info=True)

    # <<< PROCESSING LOOP FOR EACH CONVERSATION SEGMENT >>>
    if workflow.get("cut_between_events") and conversation_segments:
        logging.info(f"--- Processing {len(conversation_segments)} conversation segments ---")
        base_conversation_dir = os.path.join(output_dir, "conversation_segments") # Base dir for all segments
        # <<< Define and create central soundbites directory >>>
        central_soundbites_dir = os.path.join(output_dir, "soundbites")
        os.makedirs(central_soundbites_dir, exist_ok=True)
        relative_soundbites_dir_path = os.path.relpath(central_soundbites_dir, output_dir)
        
        for idx, segment_info in enumerate(conversation_segments):
            if stop_event and stop_event.is_set(): 
                logging.warning("Stop requested during segment processing loop.")
                break

            segment_abs_path = segment_info['path'] # Absolute path from cut_audio
            segment_rel_path = os.path.relpath(segment_abs_path, output_dir)
            logging.info(f"\n--- Processing Segment {idx} ({segment_rel_path}) --- ")

            # --- Create dedicated output directory for this segment --- 
            segment_output_dir = os.path.join(base_conversation_dir, f"segment_{idx:04d}")
            os.makedirs(segment_output_dir, exist_ok=True)
            logging.info(f"Segment output directory: {segment_output_dir}")

            # --- Initialize results for this specific segment --- 
            segment_results = {
                "segment_index": idx,
                "original_path": segment_rel_path,
                "start_original": segment_info['start'],
                "end_original": segment_info['end'],
                "duration": segment_info['duration'],
                "segment_output_dir": os.path.relpath(segment_output_dir, output_dir),
                "status": "pending",
                "demucs_vocals_path": None,
                "demucs_nonvocals_path": None,
                "annotation_path": None,
                "pass2_annotations": {},
                "transcription_path": None, # Path to main transcription file for segment
                "transcription_details": {}, # Whisper result dictionary
                "soundbites_dir": None,
                "soundbites": []
            }

            # --- Determine Audio Source for Processing (Demucs or Original) --- 
            audio_source_for_processing = segment_abs_path # Default to original segment
            segment_vocals_path_abs = None
            segment_nonvocals_path_abs = None

            if workflow.get("separate_vocals"):
                logging.info(f"Running Demucs vocal separation on segment {idx}...")
                segment_demucs_dir = os.path.join(segment_output_dir, "demucs")
                try:
                    vocals_path, nonvocals_path = separate_vocals_with_demucs(
                        input_audio=segment_abs_path, 
                        output_dir=segment_demucs_dir,
                        model=preset_config.get("vocal_separation", {}).get("model", "htdemucs")
                    )
                    if vocals_path:
                         segment_vocals_path_abs = vocals_path
                         audio_source_for_processing = segment_vocals_path_abs # Use vocals for next steps
                         segment_results["demucs_vocals_path"] = os.path.relpath(vocals_path, output_dir)
                         logging.info(f"Using separated vocals for subsequent processing: {segment_vocals_path_abs}")
                    else:
                         logging.warning(f"Demucs did not produce a vocals track for segment {idx}. Using original segment.")
                    if nonvocals_path:
                         segment_nonvocals_path_abs = nonvocals_path
                         segment_results["demucs_nonvocals_path"] = os.path.relpath(nonvocals_path, output_dir)
                except Exception as e:
                     logging.error(f"Error during Demucs separation for segment {idx}: {e}", exc_info=True)
                     logging.warning("Proceeding with original segment audio.")
            else:
                 logging.info("Vocal separation disabled for segments.")

            # --- Load Segment Audio Data (vocals or original) --- 
            segment_audio_data_np = None
            segment_sr = None
            try:
                logging.info(f"Loading segment audio for processing: {audio_source_for_processing}")
                segment_audio_data_np, segment_sr = sf.read(audio_source_for_processing, dtype='float32')
                if segment_sr != CLAP_SAMPLE_RATE: # Resample if needed for CLAP
                    logging.warning(f"Segment {idx} requires resampling from {segment_sr}Hz to {CLAP_SAMPLE_RATE}Hz. Using ffmpeg.")
                    resampled_segment_path = os.path.join(segment_output_dir, f"segment_{idx:04d}_48khz.wav")
                    cmd = ["ffmpeg", "-y", "-i", audio_source_for_processing, "-ar", str(CLAP_SAMPLE_RATE), "-acodec", "pcm_s16le", resampled_segment_path]
                    t_start_ffmpeg_seg = time.time()
                    try:
                        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        t_end_ffmpeg_seg = time.time()
                        logging.info(f"FFmpeg segment resampling successful in {t_end_ffmpeg_seg - t_start_ffmpeg_seg:.2f}s. Output: {resampled_segment_path}")
                        # Reload the resampled audio data
                        logging.info(f"Reloading resampled segment audio: {resampled_segment_path}")
                        segment_audio_data_np, segment_sr = sf.read(resampled_segment_path, dtype='float32')
                        # Update the source path for subsequent steps like Whisper if needed
                        # audio_source_for_processing = resampled_segment_path 
                        # ^^^ Keep original path for whisper for now, as whisper can handle diff SR.
                        # Only CLAP needed the strict 48kHz numpy array.
                    except subprocess.CalledProcessError as e:
                         logging.error(f"FFmpeg segment resampling failed! Return Code: {e.returncode}")
                         logging.error(f"FFmpeg stderr: {e.stderr}")
                         # If resampling fails, we cannot proceed with CLAP for this segment
                         segment_results["status"] = "error_resampling"
                         results["segments"].append(segment_results)
                         continue # Skip to next segment
                    except Exception as e_gen:
                         logging.error(f"An unexpected error occurred during ffmpeg segment resampling: {e_gen}")
                         segment_results["status"] = "error_resampling"
                         results["segments"].append(segment_results)
                         continue # Skip to next segment
                         
                if segment_audio_data_np.ndim > 1: segment_audio_data_np = segment_audio_data_np[:, 0] # Ensure mono
                logging.info(f"Segment {idx} audio loaded (Shape: {segment_audio_data_np.shape}, SR: {segment_sr})")
            except Exception as e:
                logging.error(f"Failed to load/resample segment audio {audio_source_for_processing}: {e}")
                segment_results["status"] = "error_loading_segment"
                results["segments"].append(segment_results)
                continue # Skip to next segment

            # --- Pass 2 Annotation (CLAP) --- 
            if workflow.get("annotate_segments") and clap_model_pass2 and clap_processor_pass2:
                logging.info(f"--- Running Pass 2 CLAP Annotation on segment {idx} ---")
                try:
                    sound_config = preset_config.get("sound_detection", {})
                    target_events = sound_config.get("target_prompts", DEFAULT_EVENTS)
                    threshold = sound_config.get("threshold", 0.5)
                    chunk_duration = sound_config.get("chunk_duration_s", 5.0)
                    min_gap = sound_config.get("min_duration", 1.0) # Use sound_detection key for Pass 2 NMS?
                    logging.info(f"Pass 2 Config: Threshold={threshold}, Chunk={chunk_duration}s, Min Gap={min_gap}s")
                    logging.info(f"Pass 2 Target Events: {target_events}")

                    pass2_annotations = run_clap_event_detection(
                        audio_data=segment_audio_data_np, sample_rate=segment_sr,
                        clap_model=clap_model_pass2, clap_processor=clap_processor_pass2,
                        device=processing_device, target_events=target_events,
                        threshold=threshold, chunk_duration=chunk_duration, min_gap=min_gap
                    )
                    segment_results["pass2_annotations"] = pass2_annotations
                    
                    # Save annotations
                    annotation_path = os.path.join(segment_output_dir, "annotations.json")
                    with open(annotation_path, 'w') as f: json.dump(pass2_annotations, f, indent=2)
                    segment_results["annotation_path"] = os.path.relpath(annotation_path, output_dir)
                    logging.info(f"Saved Pass 2 annotations for segment {idx} to {annotation_path}")

                except Exception as e:
                    logging.error(f"Error during Pass 2 CLAP annotation for segment {idx}: {e}", exc_info=True)
                    segment_results["status"] = "error_annotation"
            elif workflow.get("annotate_segments"):
                 logging.warning(f"Skipping Pass 2 annotation for segment {idx}: CLAP model not loaded.")

            # --- Segment Transcription (Whisper) --- 
            if workflow.get("transcribe") and whisper_model_pass2:
                logging.info(f"--- Running Whisper Transcription on segment {idx} ---")
                try:
                    transcribe_config = preset_config.get("transcription", {})
                    word_timestamps = transcribe_config.get("word_timestamps", False)
                    
                    # Note: Whisper takes file path, not audio data numpy array
                    transcription_result = whisper_model_pass2.transcribe(
                         audio_source_for_processing, 
                         word_timestamps=word_timestamps
                    )
                    segment_results["transcription_details"] = transcription_result # Store full Whisper result
                    
                    # Save transcription text
                    ts_text_path = os.path.join(segment_output_dir, "transcript.txt")
                    with open(ts_text_path, 'w', encoding='utf-8') as f:
                        f.write(f"Segment {idx} ({segment_results['start_original']:.2f}s - {segment_results['end_original']:.2f}s)\n\n")
                        f.write(transcription_result["text"])
                    segment_results["transcription_path"] = os.path.relpath(ts_text_path, output_dir)
                    logging.info(f"Saved transcription text for segment {idx} to {ts_text_path}")

                    # Save transcription JSON (optional, maybe remove if too large?)
                    # ts_json_path = os.path.join(segment_output_dir, "transcript.json")
                    # with open(ts_json_path, 'w') as f: json.dump(transcription_result, f, indent=2)
                    # segment_results["transcription_json_path"] = os.path.relpath(ts_json_path, output_dir)

                except Exception as e:
                    logging.error(f"Error during Whisper transcription for segment {idx}: {e}", exc_info=True)
                    segment_results["status"] = "error_transcription"
                    segment_results["transcription_details"] = {"error": str(e)}
            elif workflow.get("transcribe"):
                 logging.warning(f"Skipping transcription for segment {idx}: Whisper model not loaded.")

            # --- Soundbite Extraction --- 
            if workflow.get("extract_soundbites") and segment_results["pass2_annotations"]:
                logging.info(f"--- Extracting Soundbites for segment {idx} ---")
                try:
                    soundbite_config = preset_config.get("sound_detection", {}) # Re-use sound detection config for extraction params?
                    confidence_thresh = soundbite_config.get("confidence_threshold", 0.3) # Example threshold
                    min_duration = soundbite_config.get("min_bite_duration", 0.2)
                    padding = soundbite_config.get("bite_padding_ms", 150)
                    
                    # <<< Save soundbites to the central directory >>>
                    # soundbites_output_dir = os.path.join(segment_output_dir, "soundbites") # Old location
                    # Use segment filename without extension as base
                    segment_base_name = os.path.splitext(os.path.basename(segment_abs_path))[0]
                    
                    extracted_bite_paths = extract_soundbites(
                        segment_audio_path=audio_source_for_processing, # Use vocals if available
                        segment_annotations=segment_results["pass2_annotations"],
                        output_dir=central_soundbites_dir, # <<< Use central dir >>>
                        base_filename=f"seg{idx:04d}_{segment_base_name}",
                        min_duration_s=min_duration,
                        padding_ms=padding,
                        confidence_threshold=confidence_thresh
                    )
                    # <<< Store relative path to the central directory >>>
                    segment_results["soundbites_dir"] = relative_soundbites_dir_path 
                    segment_results["soundbites"] = [os.path.relpath(p, output_dir) for p in extracted_bite_paths]
                    logging.info(f"Extracted {len(extracted_bite_paths)} soundbites for segment {idx} to {central_soundbites_dir}.") # Log central dir
                except Exception as e:
                     logging.error(f"Error during soundbite extraction for segment {idx}: {e}", exc_info=True)
                     segment_results["status"] = "error_soundbites"
            elif workflow.get("extract_soundbites"):
                 logging.warning(f"Skipping soundbite extraction for segment {idx}: No Pass 2 annotations available.")

            # Update status if still pending
            if segment_results["status"] == "pending":
                 segment_results["status"] = "completed"
                 
            results["segments"].append(segment_results)
            # Clean up numpy array for the segment to free memory before next iteration
            del segment_audio_data_np
            if processing_device.type == 'cuda': torch.cuda.empty_cache()
            logging.info(f"--- Finished Processing Segment {idx} --- ")
            
    # --- Unload models after segment processing loop --- 
    if clap_model_pass2:
        del clap_model_pass2, clap_processor_pass2
        if processing_device.type == 'cuda': torch.cuda.empty_cache()
        logging.info("Pass 2 CLAP model/processor released.")
    if whisper_model_pass2:
        del whisper_model_pass2
        if processing_device.type == 'cuda': torch.cuda.empty_cache()
        logging.info("Segment Whisper model released.")

    # <<< ELSE: Handle Standard/Non-Segment-Based Workflows >>>
    elif not workflow.get("cut_between_events"): # Check if it's NOT the two-pass segment workflow
        logging.info("--- Running Standard Workflow (Whole File) ---")
        
        # --- Vocal Separation (Standard Workflow) ---
        vocals_path_abs = None
        nonvocals_path_abs = None
        if workflow.get("separate_vocals"):
            logging.info("Separating vocals from audio (Standard Workflow)...")
            demucs_output_dir = os.path.join(output_dir, "demucs") # Standard location
            vocals_path, nonvocals_path = separate_vocals_with_demucs(
                input_audio=audio_path_for_processing, 
                output_dir=demucs_output_dir,
                model=preset_config.get("vocal_separation", {}).get("model", "htdemucs") # Get model from config
            )
            vocals_path_abs = vocals_path # Keep absolute paths for potential use
            nonvocals_path_abs = nonvocals_path
            if vocals_path: results["vocal_path"] = os.path.relpath(vocals_path, output_dir)
            if nonvocals_path: results["nonvocal_path"] = os.path.relpath(nonvocals_path, output_dir)
            if stop_event and stop_event.is_set(): return {"status": "cancelled", **results}
        
        # --- Transcription (Standard Workflow) ---
        if workflow.get("transcribe"):
            logging.info("Transcribing audio (Standard Workflow)...")
            whisper_model = None # Ensure variable exists
            try:
                transcribe_config = preset_config.get("transcription", {})
                model_size = transcribe_config.get("model_size", "medium") # Use key from preset
                word_timestamps = transcribe_config.get("word_timestamps", True) # Default True for standard?
                t_whisper_load_start = time.time()
                whisper_model = whisper.load_model(model_size, device=processing_device)
                logging.info(f"Whisper model ({model_size}) loaded in {time.time() - t_whisper_load_start:.2f}s")
                
                # Determine audio source for transcription
                audio_source_for_transcription = audio_path_for_processing # Default to main audio
                if workflow.get("separate_vocals") and vocals_path_abs: # Prioritize vocals if separated
                    audio_source_for_transcription = vocals_path_abs
                    logging.info(f"Using separated vocals for transcription: {vocals_path_abs}")
                elif workflow.get("separate_vocals"):
                     logging.warning("Vocal separation enabled, but vocal track not found. Transcribing main audio.")
                
                # Check for event-guided transcription (using Pass 1 events if available)
                transcription_result = None
                if workflow.get("use_events_for_transcription") and results.get("pass1_events"):
                     # Implement logic similar to old block: filter pass1_events for 'speech'/'conversation',
                     # load segments from audio_source_for_transcription, transcribe, combine.
                     # This requires re-adding that specific logic if this flag is used by standard presets.
                     # For now, falling back to full transcription if use_events_for_transcription is True but segment logic isn't here.
                     logging.warning("'use_events_for_transcription' requested for standard workflow, but segment logic not implemented here. Falling back to full transcription.")
                     transcription_result = whisper_model.transcribe(audio_source_for_transcription, word_timestamps=word_timestamps)
                else:
                     # Standard full transcription
                     transcription_result = whisper_model.transcribe(audio_source_for_transcription, word_timestamps=word_timestamps)
                
                results["transcription"] = transcription_result # Store the full result
                
                # Save standard transcription output (e.g., text file, maybe JSON)
                standard_ts_dir = os.path.join(output_dir, "transcription_output") # Or just output_dir
                os.makedirs(standard_ts_dir, exist_ok=True)
                ts_text_path = os.path.join(standard_ts_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_transcript.txt")
                ts_json_path = os.path.join(standard_ts_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_transcript.json")
                try:
                    with open(ts_text_path, 'w', encoding='utf-8') as f: f.write(transcription_result["text"])
                    results["transcription_text_path"] = os.path.relpath(ts_text_path, output_dir)
                    with open(ts_json_path, 'w') as f: json.dump(transcription_result, f, indent=2)
                    results["transcription_json_path"] = os.path.relpath(ts_json_path, output_dir)
                except Exception as e: logging.error(f"Failed to save standard transcription results: {e}")

            except Exception as e:
                 logging.error(f"Error during standard Whisper transcription: {e}", exc_info=True)
                 results["transcription"] = {"error": str(e)}
            finally:
                 if whisper_model: # Unload model
                     del whisper_model
                     if processing_device.type == 'cuda': torch.cuda.empty_cache()
                     logging.info("Whisper model released.")
            if stop_event and stop_event.is_set(): return {"status": "cancelled", **results}

        # --- Sound Detection (Standard Workflow) ---
        # Re-add logic similar to old block (lines ~1118-1250)
        if workflow.get("detect_sounds"):
            logging.info("Detecting sounds in audio (Standard Workflow)...")
            sound_detection_input_path = audio_path_for_processing # Default
            if workflow.get("separate_vocals") and nonvocals_path_abs:
                 sound_detection_input_path = nonvocals_path_abs
                 logging.info(f"Using non-vocal track for sound detection: {nonvocals_path_abs}")
            elif workflow.get("separate_vocals"):
                 logging.warning("Vocal separation enabled, but non-vocal track not found. Using main audio for sound detection.")
            
            # Ensure 48kHz for sound detection input
            sound_detection_audio_path = sound_detection_input_path
            try: # Check and resample if needed
                info = sf.info(sound_detection_input_path)
                if info.samplerate != CLAP_SAMPLE_RATE:
                    logging.warning(f"Resampling sound detection input from {info.samplerate}Hz to {CLAP_SAMPLE_RATE}Hz using ffmpeg...")
                    sounds_output_dir_base = os.path.join(output_dir, "sounds")
                    os.makedirs(sounds_output_dir_base, exist_ok=True)
                    sd_resampled_path = os.path.join(sounds_output_dir_base, f"{os.path.splitext(os.path.basename(sound_detection_input_path))[0]}_48khz.wav")
                    cmd = ["ffmpeg", "-y", "-i", sound_detection_input_path, "-ar", str(CLAP_SAMPLE_RATE), "-acodec", "pcm_s16le", sd_resampled_path]
                    t_start_ffmpeg_sd = time.time()
                    try:
                        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        t_end_ffmpeg_sd = time.time()
                        logging.info(f"FFmpeg sound detection input resampling successful in {t_end_ffmpeg_sd - t_start_ffmpeg_sd:.2f}s. Output: {sd_resampled_path}")
                        sound_detection_audio_path = sd_resampled_path
                    except subprocess.CalledProcessError as e:
                         logging.error(f"FFmpeg sound detection input resampling failed! Return Code: {e.returncode}")
                         logging.error(f"FFmpeg stderr: {e.stderr}")
                         logging.error("Cannot proceed with sound detection.")
                         sound_detection_audio_path = None # Prevent proceeding
                    except Exception as e_gen:
                         logging.error(f"An unexpected error occurred during ffmpeg sound detection input resampling: {e_gen}")
                         sound_detection_audio_path = None # Prevent proceeding
            except Exception as e: 
                 logging.error(f"Error checking/resampling sound input: {e}")
                 sound_detection_audio_path = None

            if sound_detection_audio_path:
                # Get config
                sound_config = preset_config.get("sound_detection", {})
                target_prompts = sound_config.get("target_prompts", TARGET_SOUND_PROMPTS)
                threshold = sound_config.get("threshold", DEFAULT_CALL_THRESHOLD)
                chunk_duration_s = sound_config.get("chunk_duration_s", DEFAULT_CALL_CHUNK_DURATION)
                
                # Call detect_sound_events (requires loading CLAP again, or keeping it loaded)
                # NOTE: This re-adds the original detect_sound_events call which uses librosa internally
                # Consider refactoring detect_sound_events to use run_clap_event_detection like Pass 2
                detected_sounds = detect_sound_events(
                    audio_path=sound_detection_audio_path,
                    chunk_duration_s=chunk_duration_s,
                    threshold=threshold,
                    target_prompts=target_prompts
                )
                
                # Save results
                sounds_output_dir = os.path.join(output_dir, "sounds")
                os.makedirs(sounds_output_dir, exist_ok=True)
                sounds_output_path = os.path.join(sounds_output_dir, "sounds.json")
                try:
                    with open(sounds_output_path, 'w') as f: json.dump(detected_sounds, f, indent=2)
                    results["detected_sounds_path"] = os.path.relpath(sounds_output_path, output_dir)
                except Exception as e: logging.error(f"Failed to save sound detection results: {e}")
                results["detected_sounds"] = detected_sounds

                # --- Cutting based on SOUNDS (Standard Workflow) ---
                if workflow.get("cut_segments") and detected_sounds:
                    logging.info("Cutting audio based on detected sounds (Standard Workflow)...")
                    cut_segments_output_dir = os.path.join(output_dir, "cut_segments") # Standard location
                    cut_segments_result = cut_audio_at_detections(
                        audio_path=audio_path_for_processing, 
                        detected_events=detected_sounds,
                        output_dir=cut_segments_output_dir
                    )
                    if cut_segments_result:
                        results["cut_segments"] = [
                            {**seg, 'path': os.path.relpath(seg['path'], output_dir)}
                            for seg in cut_segments_result if seg.get('path')
                        ]
                    else: results["cut_segments"] = []
            if stop_event and stop_event.is_set(): return {"status": "cancelled", **results}
            
    else: # Handle case where cut_between_events was True but no segments were created
        if workflow.get("cut_between_events") and not conversation_segments:
             logging.warning("Segment processing skipped: No conversation segments created from Pass 1 events.")
    
    # --- Final Results Saving (Moved outside the standard workflow block) ---
    logging.info("--- Saving Final Results ---")
    
    # Generate Master Transcript (Handles both segment and standard results)
    master_transcript_path = os.path.join(output_dir, "master_transcript.txt")
    try:
        with open(master_transcript_path, "w", encoding='utf-8') as f:
            f.write(f"Master Transcript for: {os.path.basename(results['input_file'])}\\n")
            f.write(f"Preset: {results['preset']}\\n")
            f.write(f"Processed: {results['processing_time']}\\n\\n")
            
            if results.get("segments"):
                 segments_to_write = sorted(results["segments"], key=lambda x: x.get('start_original', 0))
                 f.write(f"Segments Processed: {len(segments_to_write)}\\n\\n")
                 f.write("="*20 + "\\n\\n")
                 for seg_res in segments_to_write:
                    # --- Write Segment Information to Master Transcript ---
                    segment_index = seg_res.get("segment_index", "N/A")
                    start_time = seg_res.get("start_original", 0)
                    end_time = seg_res.get("end_original", 0)
                    status = seg_res.get("status")
                    
                    f.write(f"--- Segment {segment_index} ({start_time:.2f}s - {end_time:.2f}s) ---\\n")
                    f.write(f"Status: {status}\\n")
                    
                    transcript_text = "(Transcription not available or failed)"
                    # Prioritize reading from the saved text file if it exists
                    transcript_file_rel = seg_res.get("transcription_path")
                    if transcript_file_rel:
                        transcript_file_abs = os.path.join(output_dir, transcript_file_rel)
                        if os.path.exists(transcript_file_abs):
                            try:
                                with open(transcript_file_abs, 'r', encoding='utf-8') as ts_f:
                                    # Skip the header line (line 0) and blank line (line 1)
                                    lines = ts_f.readlines()
                                    if len(lines) > 2:
                                        transcript_text = "".join(lines[2:]).strip()
                                    elif len(lines) == 1: # Only header
                                        transcript_text = "(Transcript file only contained header)"
                                    elif len(lines) == 2: # Header + blank line
                                         transcript_text = "(Transcript file contained header and blank line only)"
                                    else: # Empty file?
                                         transcript_text = "(Empty transcript file)"
                            except Exception as read_err:
                                transcript_text = f"(Error reading transcript file: {read_err})"
                        else:
                            transcript_text = "(Transcript file not found)"
                    # Fallback to text stored in results if file path missing or failed
                    elif seg_res.get("transcription_details") and isinstance(seg_res["transcription_details"], dict):
                        transcript_text = seg_res["transcription_details"].get("text", transcript_text)
                        
                    f.write(f"Transcription:\\n{transcript_text.strip()}\\n")
                    
                    # Optionally add annotation summary or soundbite info here too
                    num_soundbites = len(seg_res.get("soundbites", []))
                    f.write(f"Soundbites Extracted: {num_soundbites}\\n")
                    f.write("\\n" + "-"*20 + "\\n\\n") # Separator
                    # --- End Segment Information ---
                    
            elif results.get("transcription") and isinstance(results["transcription"], dict) and "text" in results["transcription"]:
                 f.write("Full Transcription:\\n")
                 f.write(results["transcription"]["text"]) # Write full text if no segments
            else:
                 f.write("(No transcription available)")
        results["master_transcript_path"] = os.path.relpath(master_transcript_path, output_dir)
        logging.info(f"Saved master transcript to {master_transcript_path}")
    except Exception as e:
        logging.error(f"Failed to generate master transcript: {e}", exc_info=True)

    # Save final results YAML (Handles both workflow types)
    results_path = os.path.join(output_dir, "results.yaml") 
    try:
        # Clean up potentially large data before saving YAML
        cleaned_results = results.copy()
        if "segments" in cleaned_results: # Clean segment data if present
            # Create a summary list for YAML instead of full segment data
            segment_summaries = []
            for seg_res in results.get("segments", []):
                 summary = {
                     "segment_index": seg_res.get("segment_index"),
                     "status": seg_res.get("status"),
                     "start": seg_res.get("start_original"),
                     "end": seg_res.get("end_original"),
                     "duration": seg_res.get("duration"),
                     "num_annotations": len(seg_res.get("pass2_annotations", {})), # Count annotations
                     "transcription_length": len(seg_res.get("transcription_details", {}).get("text", "")) if seg_res.get("transcription_details") else 0,
                     "num_soundbites": len(seg_res.get("soundbites", [])),
                     "output_dir": seg_res.get("segment_output_dir")
                 }
                 segment_summaries.append(summary)
            cleaned_results["segments"] = segment_summaries # Replace full data with summaries
        if "transcription" in cleaned_results: # Clean full transcription if present
             if isinstance(cleaned_results["transcription"], dict) and "text" in cleaned_results["transcription"]:
                 cleaned_results["transcription_summary"] = cleaned_results["transcription"]["text"][:200] + "..."
             del cleaned_results["transcription"]
        # Keep pass1_events summary, remove details? 
        # Keep detected_sounds summary? 

        with open(results_path, "w") as f:
            yaml.dump(cleaned_results, f, default_flow_style=False, sort_keys=False, width=120)
        logging.info(f"Saved final results YAML to {results_path}")
    except Exception as e:
        logging.error(f"Failed to save final results YAML: {e}", exc_info=True)
        
    results["status"] = "completed"
    return results

def copy_final_outputs(temp_dir, final_output_dir):
    """Copy only the final output files to the final output directory."""
    # Copy only necessary files, not intermediate ones
    files_to_copy = [
        "master_transcript.txt",
        "word_timings.json"
    ]
    
    directories_to_copy = [
        "*_transcriptions",
        "*_words",
    ]
    
    for file in files_to_copy:
        src = os.path.join(temp_dir, file)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(final_output_dir, file))
    
    import glob
    for pattern in directories_to_copy:
        for dir_path in glob.glob(os.path.join(temp_dir, pattern)):
            dir_name = os.path.basename(dir_path)
            target_dir = os.path.join(final_output_dir, dir_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # Copy all files in the directory
            for file in os.listdir(dir_path):
                src = os.path.join(dir_path, file)
                if os.path.isfile(src):
                    import shutil
                    shutil.copy2(src, os.path.join(target_dir, file))
    
    # Copy full transcripts
    for file in os.listdir(temp_dir):
        if file.endswith("_full_transcript.txt"):
            src = os.path.join(temp_dir, file)
            import shutil
            shutil.copy2(src, os.path.join(final_output_dir, file))

def main():
    parser = argparse.ArgumentParser(
        description="WhisperBite: Audio processing for transcription and speaker diarization."
    )
    parser.add_argument('--input_dir', type=str, help='Directory containing input audio files.')
    parser.add_argument('--input_file', type=str, help='Single audio file for processing.')
    parser.add_argument('--url', type=str, help='URL to download audio from.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--model', type=str, default="base", help='Whisper model to use (default: base).')
    parser.add_argument('--num_speakers', type=int, default=2, help='Number of speakers for diarization (default: 2).')
    parser.add_argument('--auto_speakers', action='store_true', help='Automatically detect optimal speaker count.')
    parser.add_argument('--enable_vocal_separation', action='store_true', help='Enable vocal separation using Demucs.')
    parser.add_argument('--enable_word_extraction', action='store_true', help='Enable extraction of individual word audio snippets.')
    parser.add_argument('--enable_second_pass', action='store_true', help='Enable second pass diarization for refinement.')
    parser.add_argument('--second_pass_min_duration', type=float, default=5.0, help='Minimum segment duration (seconds) for second pass refinement.')
    parser.add_argument('--split_stereo', action='store_true', help='Process left and right channels separately if input is stereo.')
    
    # Call detection and processing options
    parser.add_argument('--clap_chunk_duration', type=float, default=5.0, help='Chunk duration in seconds for CLAP processing.')
    parser.add_argument('--clap_threshold', type=float, default=0.7, help='Confidence threshold for CLAP call detection (0.0-1.0).')
    parser.add_argument('--clap_target_prompts', type=str, help='Comma-separated list of custom text prompts for call detection.')
    parser.add_argument('--disable_auto_cutting', action='store_true', help='Disable automatic cutting at detected call boundaries.')
    parser.add_argument('--save_cut_nonvocal_track', action='store_true', help='Save the cut non-vocal track segments.')
    
    # Output options
    parser.add_argument('--force_mono_output', action='store_true', help='Force output audio snippets to be mono.')
    parser.add_argument('--processing_mode', type=str, default="Call Analysis", 
                       choices=["Call Analysis", "Standard Processing"],
                       help='Select the overall processing workflow.')
    
    args = parser.parse_args()

    # Input validation
    if not any([args.input_dir, args.input_file, args.url]):
        logging.error("Please provide one of --input_dir, --input_file, or --url.")
        parser.print_help()
        sys.exit(1)

    # Ensure only one input type is provided
    input_args = [args.input_dir, args.input_file, args.url]
    if sum(1 for arg in input_args if arg is not None) > 1:
        logging.error("Please provide only one type of input: --input_dir, --input_file, or --url.")
        parser.print_help()
        sys.exit(1)

    # Determine input path
    input_path_for_processing = None
    if args.url:
        download_target_dir = os.path.join(args.output_dir, "downloads")
        os.makedirs(download_target_dir, exist_ok=True)
        try:
            logging.info(f"Downloading audio from {args.url} to {download_target_dir}")
            input_path_for_processing = download_audio(args.url, download_target_dir)
            if not input_path_for_processing:
                raise ValueError("download_audio failed to return a valid path.")
        except Exception as e:
            logging.error(f"Failed to download URL {args.url}: {e}")
            sys.exit(1)
    elif args.input_dir:
        input_path_for_processing = args.input_dir
    else:
        input_path_for_processing = args.input_file

    # Final path check
    if not input_path_for_processing or not os.path.exists(input_path_for_processing):
        logging.error(f"Input path does not exist or could not be determined: {input_path_for_processing}")
        sys.exit(1)

    # Process audio with the determined path
    process_audio(
        input_file=input_path_for_processing,
        output_dir=args.output_dir,
        preset_name="Default",
        preset_config={
            "name": "Default",
            "workflow": {
                "detect_events": True,
                "event_types": ["speech", "conversation"],
                "event_threshold": 0.98,
                "min_event_gap": 1.0,
                "separate_vocals": True,
                "transcribe": True,
                "use_events_for_transcription": True,
                "detect_sounds": True
            }
        },
        stop_event=None
    )

if __name__ == "__main__":
    main()
