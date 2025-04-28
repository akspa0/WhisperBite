import os
import sys
import argparse
from datetime import datetime
import logging
import json
import subprocess
from whisper import load_model
from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
from utils import sanitize_filename, download_audio, zip_results, get_media_info, get_audio_channels
from vocal_separation import separate_vocals_with_demucs
import re
from sound_detection import detect_sound_events, TARGET_SOUND_PROMPTS # Import default prompts
import yaml

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
        "-ar", "44100", "-acodec", "pcm_s16le", output_file
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
            original_end_time = segment_info['end']
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
        # Use ffmpeg to extract audio as 16-bit PCM WAV, 44.1kHz, mono
        # -vn: disable video recording
        # -acodec pcm_s16le: standard WAV audio codec
        # -ar 44100: audio sample rate
        # -ac 1: mono audio channel
        # -y: overwrite output file if exists
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            output_wav_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True) # Added text=True
        logging.info(f"Successfully extracted audio to: {output_wav_path}")
        # logging.debug(f"ffmpeg output: {result.stdout}") # Optional: log ffmpeg output
        # logging.debug(f"ffmpeg stderr: {result.stderr}")
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

def process_audio(input_path, output_dir, model_name, enable_vocal_separation, num_speakers, 
                  auto_speakers=False, enable_word_extraction=False, enable_second_pass=False,
                  attempt_sound_detection=False, second_pass_min_duration=5.0,
                  split_stereo=False, input_url=None, 
                  clap_chunk_duration: float = 5.0, 
                  clap_threshold: float = 0.7,
                  clap_target_prompts: str | None = None,
                  force_mono_output: bool = False):
    """Unified pipeline for audio processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load models FIRST to avoid repeated loading
    logging.info(f"Loading Whisper model: {model_name}")
    model = load_model(model_name)
    
    logging.info("Loading diarization pipeline")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    pipeline.to(device)
    
    # Determine initial input file and base name
    initial_input_file = None
    input_basename = None
    extracted_audio_temp_file = None # Keep track if we extract audio

    if os.path.isfile(input_path):
        initial_input_file = input_path
        input_basename = os.path.splitext(os.path.basename(initial_input_file))[0]
    elif os.path.isdir(input_path):
        # If directory, find the most recent compatible file (audio or video)
        logging.info(f"Input is a directory: {input_path}. Searching for newest compatible file.")
        compatible_extensions = ('.mp3', '.wav', '.m4a', '.ogg', '.flac') + VIDEO_EXTENSIONS
        all_files = [
            f for f in os.listdir(input_path) 
            if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith(compatible_extensions)
        ]
        if not all_files:
            logging.error(f"No compatible audio or video files found in directory: {input_path}")
            return None
            
        all_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_path, x)), reverse=True)
        newest_file = all_files[0]
        logging.info(f"Processing newest file found in directory: {newest_file}")
        initial_input_file = os.path.join(input_path, newest_file)
        input_basename = os.path.splitext(newest_file)[0]
    else:
        logging.error(f"Input path exists but is neither a file nor a directory: {input_path}")
        return None

    # Create output directory based on the original input base name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output_dir = os.path.join(output_dir, f"{input_basename}_{timestamp}") 
    os.makedirs(final_output_dir, exist_ok=True)
    logging.info(f"Output directory set to: {final_output_dir}")

    # --- Setup File Logging --- 
    log_file_path = os.path.join(final_output_dir, 'processing.log')
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s") # Use the same format
    file_handler.setFormatter(log_formatter)
    
    root_logger = logging.getLogger() # Get the root logger
    root_logger.addHandler(file_handler)
    logging.info(f"File logging started. Log file: {log_file_path}")
    # --- End File Logging Setup --- 

    # --- Parse CLAP Prompts --- 
    clap_prompts_to_use = None
    if attempt_sound_detection and clap_target_prompts:
        # Split comma-separated string, strip whitespace, remove empty strings
        custom_prompts = [p.strip() for p in clap_target_prompts.split(',') if p.strip()]
        if custom_prompts:
            clap_prompts_to_use = custom_prompts
            logging.info(f"Using custom CLAP prompts: {clap_prompts_to_use}")
        else:
            logging.warning("Custom CLAP prompts string provided but was empty after parsing. Using defaults.")
            clap_prompts_to_use = TARGET_SOUND_PROMPTS # Fallback to default
    elif attempt_sound_detection:
        clap_prompts_to_use = TARGET_SOUND_PROMPTS # Use default if detection enabled but no custom provided
        logging.info(f"Using default CLAP prompts: {clap_prompts_to_use}")
    # else: clap_prompts_to_use remains None if sound detection is off
    # --- End Parse CLAP Prompts --- 

    # --- Initialize YAML Output Structure --- 
    final_yaml_data = {
        'metadata': {
            'input_source': os.path.basename(initial_input_file) if initial_input_file else 'N/A',
            'input_url': input_url if input_url else None,
            'processing_timestamp': timestamp,
            'options': {
                'model': model_name,
                'num_speakers': num_speakers,
                'auto_speakers': auto_speakers,
                'vocal_separation': enable_vocal_separation,
                'word_extraction': enable_word_extraction,
                'second_pass': enable_second_pass,
                'second_pass_min_duration': second_pass_min_duration if enable_second_pass else None,
                'sound_detection': attempt_sound_detection,
                'split_stereo': split_stereo,
                'clap_chunk_duration': clap_chunk_duration if attempt_sound_detection else None,
                'clap_threshold': clap_threshold if attempt_sound_detection else None,
                'clap_target_prompts': clap_prompts_to_use if attempt_sound_detection else None,
                'force_mono_output': force_mono_output
            },
            'media_info': None # Will be populated later
        },
        # Segments structure will be added based on processing path
    }
    # --- End YAML Structure Init --- 

    audio_to_process = None
    try:
        # --- Get Media Info --- 
        logging.info(f"Getting media info for: {initial_input_file}")
        final_yaml_data['metadata']['media_info'] = get_media_info(initial_input_file)
        # --- End Media Info --- 
        
        # --- Video Extraction / Initial Audio Setup (Keep as is) --- 
        if initial_input_file.lower().endswith(VIDEO_EXTENSIONS):
            logging.info(f"Detected video input: {initial_input_file}")
            extracted_audio_temp_file = os.path.join(final_output_dir, f"{input_basename}_extracted_audio.wav")
            
            if not extract_audio_from_video(initial_input_file, extracted_audio_temp_file):
                logging.error("Audio extraction failed. Aborting processing.")
                raise RuntimeError("Audio extraction failed") # Raise to ensure cleanup
            
            audio_to_process = extracted_audio_temp_file
        else:
            audio_to_process = initial_input_file
        logging.info(f"Audio file for processing: {audio_to_process}")
        # --- End Video Extraction --- 

        # 1. Normalize the audio (use the file determined above)
        normalized_file = normalize_audio(audio_to_process, final_output_dir)
        logging.info(f"Normalized audio: {normalized_file}")
        
        # The rest of the pipeline uses the *normalized* file path
        pipeline_audio_input = normalized_file
        no_vocals_path = None # Initialize path for non-vocal track
        
        # 2. Optional vocal separation
        if enable_vocal_separation:
            logging.info("Attempting vocal separation")
            try:
                logging.info(f"[Vocal Separation] Calling separate_vocals_with_demucs with input: {pipeline_audio_input}")
                # Now receives a tuple: (vocals_path, no_vocals_path)
                vocals_file, no_vocals_path_returned = separate_vocals_with_demucs(pipeline_audio_input, final_output_dir)
                logging.info(f"[Vocal Separation] separate_vocals_with_demucs returned: Vocals={vocals_file}, NoVocals={no_vocals_path_returned}")
                
                # Update main pipeline input if vocals file is valid
                if vocals_file and os.path.exists(vocals_file):
                    pipeline_audio_input = vocals_file # Update input for next steps
                    logging.info(f"[Vocal Separation] Successfully updated pipeline_audio_input to: {pipeline_audio_input}")
                else:
                    logging.warning("Vocal separation failed or produced no output, using normalized audio for subsequent steps.")
                
                # Store the no_vocals path if valid
                if no_vocals_path_returned and os.path.exists(no_vocals_path_returned):
                    no_vocals_path = no_vocals_path_returned
                    logging.info(f"[Vocal Separation] Found valid no_vocals track: {no_vocals_path}")
                else:
                    logging.info("[Vocal Separation] No valid no_vocals track found or returned.")
                    no_vocals_path = None # Ensure it's None

            except Exception as e:
                logging.warning(f"Vocal separation error: {e}. Using normalized audio for subsequent steps.")
                vocals_file = None # Ensure these are None on error
                no_vocals_path = None
        
        # --- Check Channels for Stereo Split Logic --- 
        channels = get_audio_channels(pipeline_audio_input)
        perform_stereo_split = split_stereo and channels == 2
        logging.info(f"Audio has {channels} channels. Stereo split active: {perform_stereo_split}")
        # --- End Channel Check --- 

        # --- Initialize Segment Lists --- 
        first_pass_segments_L = []
        first_pass_segments_R = []
        first_pass_segments_mono = []
        sound_event_segments = []
        # --- End Init --- 
        
        # --- Define and create speaker output directory (needed for both paths) ---
        speaker_output_dir = os.path.join(final_output_dir, "speakers") 
        os.makedirs(speaker_output_dir, exist_ok=True)
        # --- End Directory Definition ---
        
        # --- Main Processing Logic (Conditional Stereo Split) --- 
        if perform_stereo_split:
            logging.info("Processing stereo channels separately.")
            stereo_processing_dir = os.path.join(final_output_dir, "stereo_split")
            os.makedirs(stereo_processing_dir, exist_ok=True)
            
            # Split into L/R
            try:
                logging.info(f"Splitting stereo file: {pipeline_audio_input}")
                stereo_audio = AudioSegment.from_file(pipeline_audio_input)
                mono_channels = stereo_audio.split_to_mono()
                if len(mono_channels) == 2:
                    left_channel_path = os.path.join(stereo_processing_dir, f"{input_basename}_L.wav")
                    right_channel_path = os.path.join(stereo_processing_dir, f"{input_basename}_R.wav")
                    mono_channels[0].export(left_channel_path, format="wav")
                    mono_channels[1].export(right_channel_path, format="wav")
                    logging.info(f"Exported Left channel to: {left_channel_path}")
                    logging.info(f"Exported Right channel to: {right_channel_path}")
                else:
                    logging.warning(f"Expected 2 channels after splitting {pipeline_audio_input}, but got {len(mono_channels)}. Proceeding with mono.")
                    perform_stereo_split = False # Fallback to mono processing
            except Exception as split_err:
                logging.error(f"Error splitting stereo audio: {split_err}. Proceeding with mono.")
                perform_stereo_split = False # Fallback to mono processing

            # If split succeeded, process L and R
            if perform_stereo_split:
                # --- Process Left Channel --- 
                logging.info("--- Processing Left Channel ---")
                actual_num_speakers_L = num_speakers # TODO: Auto-detect per channel?
                if auto_speakers:
                     try:
                         actual_num_speakers_L = detect_optimal_speakers(pipeline, left_channel_path, max_speakers=max(1, num_speakers))
                     except Exception as e:
                         logging.warning(f"Auto speaker detection failed for Left channel: {e}. Using provided value: {num_speakers}")
                         actual_num_speakers_L = num_speakers
                
                logging.info(f"Running L-channel diarization with {actual_num_speakers_L} speakers")
                diarization_L = pipeline(left_channel_path, num_speakers=actual_num_speakers_L)
                segment_info_dict_L = slice_audio_by_speaker(left_channel_path, diarization_L, speaker_output_dir, speaker_suffix="_L", force_mono_output=force_mono_output) # Pass flag
                logging.info(f"Sliced L-channel audio: {list(segment_info_dict_L.keys())}")
                first_pass_segments_L = transcribe_with_whisper(model, segment_info_dict_L, final_output_dir, enable_word_extraction=enable_word_extraction, speaker_suffix="_L", force_mono_output=force_mono_output) # Pass flag
                logging.info("L-channel transcription complete")
                # --- End Process Left Channel --- 
                
                # --- Process Right Channel --- 
                logging.info("--- Processing Right Channel ---")
                actual_num_speakers_R = num_speakers # TODO: Auto-detect per channel?
                if auto_speakers:
                    try:
                         actual_num_speakers_R = detect_optimal_speakers(pipeline, right_channel_path, max_speakers=max(1, num_speakers))
                    except Exception as e:
                         logging.warning(f"Auto speaker detection failed for Right channel: {e}. Using provided value: {num_speakers}")
                         actual_num_speakers_R = num_speakers
                         
                logging.info(f"Running R-channel diarization with {actual_num_speakers_R} speakers")
                diarization_R = pipeline(right_channel_path, num_speakers=actual_num_speakers_R)
                segment_info_dict_R = slice_audio_by_speaker(right_channel_path, diarization_R, speaker_output_dir, speaker_suffix="_R", force_mono_output=force_mono_output) # Pass flag
                logging.info(f"Sliced R-channel audio: {list(segment_info_dict_R.keys())}")
                first_pass_segments_R = transcribe_with_whisper(model, segment_info_dict_R, final_output_dir, enable_word_extraction=enable_word_extraction, speaker_suffix="_R", force_mono_output=force_mono_output) # Pass flag
                logging.info("R-channel transcription complete")
                # --- End Process Right Channel --- 
        
        # --- Mono Processing Path (Default or Fallback) ---
        if not perform_stereo_split:
            logging.info("Processing as single (mono) track.")
            # 3. Diarization
            actual_num_speakers = num_speakers
            if auto_speakers:
                try:
                    actual_num_speakers = detect_optimal_speakers(pipeline, pipeline_audio_input)
                except Exception as e:
                    logging.warning(f"Auto speaker detection failed: {e}. Using provided value: {num_speakers}")
                    actual_num_speakers = num_speakers
            
            logging.info(f"Running diarization with {actual_num_speakers} speakers on {pipeline_audio_input}")
            diarization = pipeline(pipeline_audio_input, num_speakers=actual_num_speakers)
            
            # 4. Slicing
            segment_info_dict = slice_audio_by_speaker(pipeline_audio_input, diarization, speaker_output_dir, force_mono_output=force_mono_output) # Pass flag
            logging.info(f"Sliced audio into speakers: {list(segment_info_dict.keys())}")
            
            # 5. Transcription
            logging.info("Starting transcription")
            first_pass_segments_mono = transcribe_with_whisper(model, segment_info_dict, final_output_dir, enable_word_extraction=enable_word_extraction, force_mono_output=force_mono_output) # Pass flag
            logging.info("Transcription complete")

            # 5b. Optional Second Pass (Only for Mono Path for now)
            final_segments_for_transcript = first_pass_segments_mono # Default to first pass
            refined_segments_info = [] 
            if enable_second_pass:
                logging.info("Starting second pass refinement...")
                refined_segments_info = run_second_pass_diarization(
                    first_pass_segment_info=segment_info_dict, 
                    first_pass_output_dir=speaker_output_dir, 
                    diarization_pipeline=pipeline,
                    whisper_model=model,
                    final_output_dir=final_output_dir, 
                    segment_min_duration=second_pass_min_duration,
                    force_mono_output=force_mono_output
                )
                logging.info("Second pass refinement complete.")
            
                # --- Merge first and second pass results (for mono path) --- 
                if refined_segments_info:
                    logging.info("Merging first and second pass transcripts for mono processing...")
                    # Create lookup dict as before
                    refined_lookup = {}
                    parsing_failures = 0 
                    for ref_seg in refined_segments_info:
                        try:
                            original_seq = int(os.path.basename(ref_seg['audio_file']).split('_')[0])
                            if original_seq not in refined_lookup:
                                refined_lookup[original_seq] = []
                            refined_lookup[original_seq].append(ref_seg)
                        except (IndexError, ValueError):
                            logging.warning(f"[Merge Debug] PARSE FAILED for original sequence from refined segment filename: {ref_seg['audio_file']}") 
                            parsing_failures += 1
                            continue
                    if parsing_failures > 0:
                        logging.warning(f"[Merge Debug] Total filename parsing failures: {parsing_failures}")

                    merged_segments = []
                    processed_first_pass_seqs = set(refined_lookup.keys())
                    
                    # Add non-refined first pass segments
                    for fp_seg in first_pass_segments_mono:
                        if fp_seg['sequence'] not in processed_first_pass_seqs:
                            merged_segments.append(fp_seg)
                    
                    # Add all refined segments
                    for ref_list in refined_lookup.values():
                        merged_segments.extend(ref_list)
                        
                    merged_segments.sort(key=lambda x: x['start'])
                    final_segments_for_transcript = merged_segments
                    logging.info("Mono merging complete.")
                else:
                    logging.info("No segments were refined in second pass for mono processing.")
                    # Keep final_segments_for_transcript as first_pass_segments_mono
            else:
                logging.info("Second pass refinement skipped for mono processing.")
                # Keep final_segments_for_transcript as first_pass_segments_mono
        # --- End Mono Processing Path --- 

        # --- Sound Detection (Run on no_vocals regardless of split) --- 
        if enable_vocal_separation and attempt_sound_detection and no_vocals_path:
            logging.info(f"Attempting sound detection with CLAP on non-vocal track: {no_vocals_path}")
            logging.info(f"CLAP Params: Chunk={clap_chunk_duration}s, Threshold={clap_threshold}")
            # Logging of prompts to use is handled during parsing above
            try:
                sound_event_segments = detect_sound_events(
                    audio_path=no_vocals_path, 
                    chunk_duration_s=clap_chunk_duration, 
                    threshold=clap_threshold,
                    target_prompts=clap_prompts_to_use # Pass the parsed list (or None for default)
                )
                logging.info(f"CLAP sound detection finished. Found {len(sound_event_segments)} potential sound events.")
                # --- Log Individual Events --- 
                if sound_event_segments:
                    logging.info("--- Detected CLAP Sound Events --- ")
                    for event in sound_event_segments:
                        logging.info(f"  - Label: \"{event.get('text', 'N/A')}\" | Time: {event.get('start', 0.0):.2f}s - {event.get('end', 0.0):.2f}s | Confidence: {event.get('confidence', 0.0):.3f}")
                    logging.info("----------------------------------")
                # --- End Log Individual Events --- 
            except Exception as sound_err:
                logging.error(f"Error during CLAP sound detection process for {no_vocals_path}: {sound_err}", exc_info=True)
                sound_event_segments = [] 
        else:
            logging.info("Sound detection skipped based on options or missing no_vocals track.")
        # --- End Sound Detection --- 

        # --- Prepare Final YAML Data Structure --- 
        logging.info("Preparing final data structure for YAML output...")
        
        # Make paths relative for YAML output
        def make_path_relative(absolute_path):
            if absolute_path and os.path.exists(absolute_path):
                # Calculate path relative to the *parent* of final_output_dir
                # because the zip file is created there.
                output_parent_dir = os.path.dirname(final_output_dir)
                try:
                    relative = os.path.relpath(absolute_path, output_parent_dir)
                    # Use forward slashes for better cross-platform compatibility in YAML/zip
                    return relative.replace("\\", "/") 
                except ValueError:
                    # Paths might be on different drives on Windows
                    logging.warning(f"Could not make path relative (different drives?): {absolute_path}")
                    return absolute_path # Fallback to absolute path
            return None

        if perform_stereo_split:
            # Structure for separate L/R channels
            final_yaml_data['left_channel_segments'] = [
                {**seg, 'audio_file': make_path_relative(seg.get('audio_file')),
                        'transcript_file': make_path_relative(seg.get('transcript_file'))}
                for seg in sorted(first_pass_segments_L, key=lambda x: x['start'])
            ]
            final_yaml_data['right_channel_segments'] = [
                 {**seg, 'audio_file': make_path_relative(seg.get('audio_file')),
                         'transcript_file': make_path_relative(seg.get('transcript_file'))}
                 for seg in sorted(first_pass_segments_R, key=lambda x: x['start'])
            ]
        else:
            # Structure for mono processing (potentially with nested refinements)
            # Rework the merging logic slightly to build the nested structure directly
            processed_segments_for_yaml = []
            processed_fp_sequences = set()
            
            # Add refined segments first, storing them under their original
            if enable_second_pass and refined_segments_info:
                 refined_lookup_yaml = {} # Build lookup {orig_seq: [refined_segment_dicts]} 
                 for ref_seg in refined_segments_info:
                     # ... (parsing logic to get original_seq) ...
                     try:
                         original_seq = int(os.path.basename(ref_seg['audio_file']).split('_')[0])
                         if original_seq not in refined_lookup_yaml:
                             refined_lookup_yaml[original_seq] = []
                         # Make paths relative here
                         ref_seg_relative = {**ref_seg,
                                              'audio_file': make_path_relative(ref_seg.get('audio_file')),
                                              'transcript_file': make_path_relative(ref_seg.get('transcript_file'))}
                         refined_lookup_yaml[original_seq].append(ref_seg_relative)
                     except (IndexError, ValueError):
                         continue # Skip segments where original sequence can't be parsed
                 
                 # Iterate through first pass, adding original + nested refined segments
                 for fp_seg in first_pass_segments_mono:
                     if fp_seg['sequence'] in refined_lookup_yaml:
                         processed_fp_sequences.add(fp_seg['sequence'])
                         processed_segments_for_yaml.append({
                             **fp_seg,
                             'audio_file': make_path_relative(fp_seg.get('audio_file')),
                             'transcript_file': make_path_relative(fp_seg.get('transcript_file')),
                             'refined': True,
                             'refined_segments': sorted(refined_lookup_yaml[fp_seg['sequence']], key=lambda x: x['start'])
                         })
                     # else: # Handled in next loop
                     #     pass
            
            # Add remaining unrefined first-pass segments
            for fp_seg in first_pass_segments_mono:
                if fp_seg['sequence'] not in processed_fp_sequences:
                     processed_segments_for_yaml.append({
                         **fp_seg,
                         'audio_file': make_path_relative(fp_seg.get('audio_file')),
                         'transcript_file': make_path_relative(fp_seg.get('transcript_file')),
                         'refined': False
                     })
            
            # Sort the final list of segments (originals, potentially with nested refined ones)
            processed_segments_for_yaml.sort(key=lambda x: x['start'])
            final_yaml_data['segments'] = processed_segments_for_yaml

        # Add sound events (paths are already None, structure adjusted slightly in CLAP func)
        final_yaml_data['sound_events'] = sorted(sound_event_segments, key=lambda x: x['start'])
        logging.info("Data structure preparation complete.")
        # --- End YAML Data Prep --- 
        
        # --- Write YAML Output --- 
        yaml_output_path = os.path.join(final_output_dir, "master_transcript.yaml")
        logging.info(f"Writing final YAML output to: {yaml_output_path}")
        try:
            with open(yaml_output_path, 'w', encoding='utf-8') as f:
                # Use SafeDumper for security if loading untrusted YAML later
                yaml.dump(final_yaml_data, f, indent=2, allow_unicode=True, sort_keys=False, Dumper=yaml.SafeDumper)
        except Exception as yaml_err:
            logging.error(f"Failed to write YAML output: {yaml_err}")
            # Fallback: maybe write a JSON dump?
            json_fallback_path = os.path.join(final_output_dir, "master_transcript_fallback.json")
            try:
                 with open(json_fallback_path, 'w', encoding='utf-8') as jf:
                     json.dump(final_yaml_data, jf, indent=2)
                 logging.warning(f"YAML write failed, wrote JSON fallback to: {json_fallback_path}")
            except Exception as json_err:
                 logging.error(f"Failed to write JSON fallback as well: {json_err}")
        # --- End YAML Output --- 

        # 6. Zip results (adjust to include YAML)
        zip_file = zip_results(final_output_dir, initial_input_file) 
        logging.info(f"Results zipped to: {zip_file}")
        
        return final_output_dir
        
    except Exception as e:
        # Catch general processing errors 
        logging.error(f"Error processing {initial_input_file}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        # Ensure YAML data includes the error if possible before returning None
        final_yaml_data['error'] = traceback.format_exc()
        yaml_output_path = os.path.join(final_output_dir, "master_transcript_error.yaml")
        try:
             with open(yaml_output_path, 'w', encoding='utf-8') as f:
                 yaml.dump(final_yaml_data, f, indent=2, allow_unicode=True, sort_keys=False, Dumper=yaml.SafeDumper)
        except Exception as final_yaml_err:
             logging.error(f"Failed to write error YAML: {final_yaml_err}")
        return None # Indicate failure
    finally:
        # Cleanup: Remove temporary extracted audio file
        if extracted_audio_temp_file and os.path.exists(extracted_audio_temp_file):
            try:
                 logging.info(f"Cleaning up temporary extracted audio file: {extracted_audio_temp_file}")
                 os.remove(extracted_audio_temp_file)
            except OSError as e:
                 logging.warning(f"Could not remove temporary audio file {extracted_audio_temp_file}: {e}")
        
        # --- Clean up File Logging (Keep as is) --- 
        if file_handler:
            logging.info("Stopping file logging.")
            root_logger.removeHandler(file_handler)
            file_handler.close()
        # --- End File Logging Cleanup ---

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
    parser.add_argument('--auto_speakers', action='store_true', help='Automatically detect optimal speaker count (Default: Disabled).')
    parser.add_argument('--enable_vocal_separation', action='store_true', help='Enable vocal separation using Demucs.')
    parser.add_argument('--enable_word_extraction', action='store_true', help='Enable extraction of individual word audio snippets.')
    parser.add_argument('--enable_second_pass', action='store_true', help='Enable second pass diarization for refinement (experimental).')
    parser.add_argument('--second_pass_min_duration', type=float, default=5.0, help='Minimum segment duration (seconds) to consider for second pass refinement (default: 5.0).')
    parser.add_argument('--split_stereo', action='store_true', help='Process left and right channels separately if input is stereo.')
    parser.add_argument('--attempt_sound_detection', action='store_true', help='Enable sound detection using CLAP (requires --enable_vocal_separation).')
    parser.add_argument('--clap_chunk_duration', type=float, default=5.0, help='Chunk duration in seconds for CLAP processing (default: 5.0).')
    parser.add_argument('--clap_threshold', type=float, default=0.7, help='Confidence threshold for CLAP sound detection (0.0-1.0, default: 0.7).')
    parser.add_argument('--clap_target_prompts', type=str, default=None, help='Comma-separated list of custom text prompts for CLAP sound detection (e.g., "dial tone,telephone ringing"). Overrides defaults.')
    parser.add_argument('--force_mono_output', action='store_true', help='Force output audio snippets (speaker segments, words) to be mono.')
    args = parser.parse_args()

    # Restore original input validation
    if not any([args.input_dir, args.input_file, args.url]):
        logging.error("Please provide one of --input_dir, --input_file, or --url.")
        parser.print_help()
        sys.exit(1)
    # Ensure only one input type is provided (optional, but good practice)
    input_args = [args.input_dir, args.input_file, args.url]
    if sum(1 for arg in input_args if arg is not None) > 1:
         logging.error("Please provide only one type of input: --input_dir, --input_file, or --url.")
         parser.print_help()
         sys.exit(1)

    # Restore original input path determination logic
    # Specific output dir determination remains inside process_audio
    input_path_for_processing = None
    if args.url:
        # For CLI, download first to a temp/output location, then pass the file path.
        download_target_dir = os.path.join(args.output_dir, "downloads") # Download to a subdir
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
        input_path_for_processing = args.input_dir # Pass the directory path
    else: # Must be input_file
        input_path_for_processing = args.input_file

    # Final check on the path before processing
    if not input_path_for_processing or not os.path.exists(input_path_for_processing):
        logging.error(f"Input path does not exist or could not be determined: {input_path_for_processing}")
        sys.exit(1)
    # No isfile check here, allow directories to be passed

    # Call process_audio with the determined path and the main output dir
    process_audio(
        input_path=input_path_for_processing, 
        output_dir=args.output_dir,
        model_name=args.model,
        enable_vocal_separation=args.enable_vocal_separation,
        num_speakers=args.num_speakers,
        auto_speakers=args.auto_speakers,
        enable_word_extraction=args.enable_word_extraction,
        enable_second_pass=args.enable_second_pass,
        second_pass_min_duration=args.second_pass_min_duration,
        attempt_sound_detection=args.attempt_sound_detection,
        split_stereo=args.split_stereo,
        input_url=args.url,
        clap_chunk_duration=args.clap_chunk_duration,
        clap_threshold=args.clap_threshold,
        clap_target_prompts=args.clap_target_prompts,
        force_mono_output=args.force_mono_output
    )

if __name__ == "__main__":
    main()
