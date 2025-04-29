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

# Third-party imports
import torch
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, ClapModel

# Local imports
from utils import sanitize_filename, download_audio, zip_results, get_media_info, get_audio_channels
from vocal_separation import separate_vocals_with_demucs, enhance_vocals, export_audio_segments
from sound_detection import (
    detect_sound_events,
    detect_and_cut_audio,
    cut_audio_at_detections,
    TARGET_SOUND_PROMPTS,
    DEFAULT_CALL_CHUNK_DURATION,
    DEFAULT_CALL_THRESHOLD,
    # CALL_ANALYSIS_PROMPTS, # Not currently used directly here
    # VOCAL_CUES_FOR_ANALYSIS, # Not currently used directly here
)
from event_detection import EventDetector, detect_and_save_events, DEFAULT_EVENTS

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
    Process audio file according to preset configuration.
    
    Args:
        input_file: Path to input audio/video file
        output_dir: Directory to save outputs
        preset_name: Name of the preset being used
        preset_config: Processing preset configuration (dict inside the full preset)
        stop_event: Optional threading.Event for cancellation
        device: Device to run models on
        
    Returns:
        Dict containing processing results and metadata
    """
    if stop_event and stop_event.is_set():
        return {"status": "cancelled"}
        
    workflow = preset_config.get("workflow", {}) 
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and normalize audio
    audio_path = get_audio_from_input(input_file, output_dir)
    
    results = {
        "input_file": input_file,
        "output_dir": output_dir,
        "preset": preset_name,
        "audio_path": audio_path,
        "processing_time": datetime.datetime.now().isoformat(),
        "device": device
    }
    
    # Event detection (if enabled)
    if workflow.get("detect_events"):
        logging.info("Detecting events in audio...")
        
        # Get config, using defaults if sections or keys are missing
        event_config = preset_config.get("event_detection", {})
        target_events = event_config.get("target_events") # Get configured events
        threshold = event_config.get("threshold", 0.98) # Use config threshold or default
        # Use min_duration from config as it's named there, default to 1.0
        min_gap = event_config.get("min_duration", 1.0) 

        # Use default prompts if the config list is empty or None
        if not target_events:
            logging.warning("No target events specified in config, using default events.")
            target_events = DEFAULT_EVENTS # Use imported defaults
            
        logging.info(f"Using target events: {target_events}") # Log which events are used

        event_results = detect_and_save_events(
            audio_path=audio_path,
            output_dir=output_dir,
            target_events=target_events, # Pass the final (possibly defaulted) list
            threshold=threshold,
            min_gap=min_gap # Pass the correctly retrieved min_gap
        )

        results["events"] = event_results
        
        if stop_event and stop_event.is_set():
            return {"status": "cancelled", **results}
    
    # Vocal separation (if enabled)
    if workflow.get("separate_vocals"):
        logging.info("Separating vocals from audio...")
        
        # First separate vocals using demucs
        vocals_path, nonvocals_path = separate_vocals_with_demucs(
            input_audio=audio_path,
            output_dir=output_dir
        )
        
        if vocals_path and nonvocals_path:
            # Create audio segments
            vocals = AudioSegment.from_file(vocals_path)
            nonvocals = AudioSegment.from_file(nonvocals_path)
            
            # Export segments
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            vocal_path, nonvocal_path = export_audio_segments(
                audio_segments=[vocals, nonvocals],
                output_dir=output_dir,
                base_name=base_name
            )
            
            results["vocal_path"] = vocal_path
            results["nonvocal_path"] = nonvocal_path
        
        if stop_event and stop_event.is_set():
            return {"status": "cancelled", **results}
    
    # Transcription (if enabled)
    if workflow.get("transcribe"):
        logging.info("Transcribing audio...")
        
        # Load Whisper model
        model = whisper.load_model("medium", device=device)
        
        # Use events to guide transcription if available
        if workflow.get("use_events_for_transcription") and "events" in results and results["events"]:
            segments = []
            logging.info("Using detected events to guide transcription.")
            # results["events"] is Dict[str, List[Dict]]
            # Iterate through the lists of detections for relevant event types
            for event_type in ["speech", "conversation"]:
                if event_type in results["events"]:
                    for event in results["events"][event_type]: # Iterate list of detections
                        try:
                            # Check event structure (basic)
                            if not all(k in event for k in ['start', 'end', 'type', 'confidence']):
                                logging.warning(f"Skipping malformed event dict: {event}")
                                continue
                                
                            # event["type"] should match event_type, but double-check
                            if event["type"] == event_type: 
                                # Transcribe speech segments
                                segment_start = event["start"]
                                segment_end = event["end"]
                                
                                # Ensure start/end are valid numbers and end > start
                                if not (isinstance(segment_start, (int, float)) and 
                                        isinstance(segment_end, (int, float)) and 
                                        segment_end > segment_start):
                                    logging.warning(f"Skipping event with invalid times: start={segment_start}, end={segment_end}")
                                    continue

                                logging.debug(f"Transcribing segment for event: {event_type} from {segment_start:.2f}s to {segment_end:.2f}s")

                                # Load audio segment safely
                                try:
                                    # Add a small buffer in case times are exact boundaries
                                    audio_segment = AudioSegment.from_file(audio_path)[
                                        int(segment_start * 1000 - 10):int(segment_end * 1000 + 10)
                                    ]
                                except Exception as load_err:
                                     logging.error(f"Error loading audio segment ({segment_start:.2f}s-{segment_end:.2f}s): {load_err}")
                                     continue # Skip this event if audio fails to load

                                # Convert to numpy array
                                # Ensure mono for Whisper if needed (model expects mono)
                                if audio_segment.channels > 1:
                                    audio_segment = audio_segment.set_channels(1)
                                samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
                                
                                # Transcribe segment
                                result = model.transcribe(samples)
                                transcript_text = result["text"].strip()

                                if transcript_text:
                                    segments.append({
                                        "start": segment_start,
                                        "end": segment_end,
                                        "text": transcript_text,
                                        "source_event_type": event_type,
                                        "source_event_confidence": event['confidence']
                                    })
                                else:
                                    logging.debug(f"Segment {segment_start:.2f}s-{segment_end:.2f}s resulted in empty transcription.")
                        except Exception as e:
                            logging.error(f"Error processing event {event}: {e}", exc_info=True)
                            continue # Skip to next event on error
                else:
                     logging.info(f"No events of type '{event_type}' found for guided transcription.")

            if segments:
                 # Sort final segments by start time
                 segments.sort(key=lambda x: x['start'])
                 results["transcription"] = {
                     "segments": segments,
                     "text": " ".join(seg["text"] for seg in segments)
                 }
                 logging.info(f"Generated {len(segments)} transcript segments based on detected events.")
            else:
                 logging.warning("Event-guided transcription enabled, but no valid speech/conversation events found or transcribed. Falling back to full transcription.")
                 # Fallback to full transcription if no segments were generated
                 result = model.transcribe(audio_path)
                 results["transcription"] = result

        else:
            # Regular full transcription (if not event-guided or no events found)
            # Check if transcription already exists from fallback
            if "transcription" not in results:
                logging.info("Performing full transcription.")
                result = model.transcribe(audio_path)
                results["transcription"] = result
            else:
                 logging.info("Skipping full transcription as event-guided fallback already ran.")

        if stop_event and stop_event.is_set():
            return {"status": "cancelled", **results}
    
    # Sound detection (if enabled)
    if workflow.get("detect_sounds"):
        logging.info("Detecting sounds in audio...")

        # Determine audio path (use original or separated vocals)
        sound_detection_audio_path = audio_path
        if workflow.get("separate_vocals") and results.get("vocal_path"):
            # Optional: Could add logic to detect on non-vocals if desired
            sound_detection_audio_path = results["vocal_path"] 
            logging.info(f"Running sound detection on separated vocal track: {sound_detection_audio_path}")
        else:
            logging.info(f"Running sound detection on original audio track: {sound_detection_audio_path}")

        # Get config for sound detection
        sound_config = preset_config.get("sound_detection", {})
        target_prompts = sound_config.get("target_prompts") # Use the key set in presets.py
        threshold = sound_config.get("threshold", DEFAULT_CALL_THRESHOLD) # Use imported default
        chunk_duration_s = sound_config.get("chunk_duration_s", DEFAULT_CALL_CHUNK_DURATION) # Use imported default

        # Default prompts if needed
        if not target_prompts:
            logging.warning("No target prompts specified for sound detection, using defaults.")
            target_prompts = TARGET_SOUND_PROMPTS # Use imported default

        logging.info(f"Using sound detection prompts: {target_prompts}")

        detected_sounds = detect_sound_events(
            audio_path=sound_detection_audio_path,
            chunk_duration_s=chunk_duration_s,
            threshold=threshold,
            target_prompts=target_prompts # Pass the final list
        )

        results["detected_sounds"] = detected_sounds
        
        # Cut audio at detected sounds
        if detected_sounds:
            cut_segments = cut_audio_at_detections(
                audio_path=audio_path,
                detected_events=detected_sounds,
                output_dir=output_dir
            )
            
            results["cut_segments"] = cut_segments
            
        if stop_event and stop_event.is_set():
            return {"status": "cancelled", **results}
    
    # Save results
    results_path = os.path.join(output_dir, "results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f)
        
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
