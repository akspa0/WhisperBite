import gradio as gr
import os
import json
from whisperBite import process_audio
import tempfile
import shutil
from utils import download_audio
import logging # Add logging
import traceback # Add traceback
import re

# Configure basic logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_pipeline(input_file, input_folder, url, output_folder, model, num_speakers, 
                auto_speakers, enable_vocal_separation, 
                enable_word_extraction, enable_second_pass, 
                second_pass_min_duration,
                attempt_sound_detection,
                hf_token,
                split_stereo,
                clap_chunk_duration,
                clap_threshold,
                clap_target_prompts,
                force_mono_output):
    """Run the audio processing pipeline based on user inputs."""
    logging.info("Starting pipeline run...")
    # Set Hugging Face token if provided
    if hf_token:
        logging.info("Setting HF_TOKEN environment variable.")
        os.environ['HF_TOKEN'] = hf_token
    else:
        logging.warning("HF_TOKEN not provided. Diarization may fail.")
        # Optionally return an error early if the token is strictly required
        # return "Error: Hugging Face Token is required for diarization.", None, ""

    if not os.path.exists(output_folder):
        logging.info(f"Creating output directory: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)

    # Determine input source (Restore folder logic)
    input_path = None
    source_type = ""
    if url:
        logging.info(f"Processing URL: {url}")
        try:
            # Use a temporary directory within the output folder for downloads
            download_dir = os.path.join(output_folder, "downloads")
            os.makedirs(download_dir, exist_ok=True)
            input_path = download_audio(url, download_dir)
            source_type = "URL"
            if not input_path:
                 raise ValueError("Download failed or returned no path.")
            logging.info(f"Downloaded audio to: {input_path}")
        except Exception as download_err:
            logging.error(f"Error downloading URL {url}: {download_err}")
            return f"Error downloading URL: {str(download_err)}", None, ""
    elif input_file is not None:
        input_path = input_file.name # Use .name attribute for Gradio File component
        logging.info(f"Processing uploaded file: {input_path}")
        source_type = "File"
    elif input_folder and os.path.isdir(input_folder):
        input_path = input_folder
        logging.info(f"Processing folder: {input_path}")
        source_type = "Folder"
    elif input_folder: # Handle case where input_folder is provided but not a valid directory
        logging.error(f"Input folder path is not a valid directory: {input_folder}")
        return f"Error: Input folder path is not a valid directory: {input_folder}", None, ""
    else:
        logging.warning("No valid input provided (file, folder, or URL).")
        return "Please provide an input file, folder, or URL.", None, "" 

    if not input_path:
        logging.error("Input path could not be determined.")
        return "Error determining input path.", None, ""
        
    logging.info(f"Final input path for processing: {input_path}")

    # Run the processing pipeline
    try:
        logging.info(f"Calling process_audio with options: model={model}, num_speakers={num_speakers}, auto={auto_speakers}, separation={enable_vocal_separation}, words={enable_word_extraction}, second_pass={enable_second_pass}, sound_detect={attempt_sound_detection}, split_stereo={split_stereo}, clap_chunk={clap_chunk_duration}, clap_thresh={clap_threshold}, force_mono={force_mono_output}")
        # Pass the new arguments
        result_dir = process_audio(
            input_path=input_path,
            output_dir=output_folder,
            model_name=model,
            enable_vocal_separation=enable_vocal_separation,
            num_speakers=num_speakers,
            auto_speakers=auto_speakers,
            enable_word_extraction=enable_word_extraction,
            enable_second_pass=enable_second_pass,
            second_pass_min_duration=second_pass_min_duration,
            attempt_sound_detection=attempt_sound_detection,
            split_stereo=split_stereo,
            clap_chunk_duration=clap_chunk_duration,
            clap_threshold=clap_threshold,
            clap_target_prompts=clap_target_prompts,
            force_mono_output=force_mono_output,
            input_url=url if source_type == "URL" else None
        )

        if not result_dir or not os.path.isdir(result_dir):
            logging.error(f"process_audio did not return a valid directory path. Got: {result_dir}")
            # Restore folder-specific fallback logic (though whisperBite currently only processes one file)
            if source_type == "Folder":
                 # This fallback might be inaccurate if whisperBite changes, but restores previous behavior
                 result_dir = output_folder 
                 logging.warning(f"process_audio returned invalid path for folder input. Assuming results are in main output folder: {result_dir}")
            else:
                 return "Processing finished, but the result directory was not found.", None, ""

        logging.info(f"Processing finished. Looking for results in: {result_dir}")

        # Find the results zip file and transcript within the specific result directory
        result_zip_file = None
        transcript = ""
        master_transcript_path = os.path.join(result_dir, "master_transcript.txt")
        second_pass_transcript_path = os.path.join(result_dir, "2nd_pass", "master_transcript.txt")

        # Prefer second pass transcript if it exists
        transcript_path_to_read = None
        if enable_second_pass and os.path.exists(second_pass_transcript_path):
            transcript_path_to_read = second_pass_transcript_path
            logging.info(f"Using second pass transcript: {transcript_path_to_read}")
        elif os.path.exists(master_transcript_path):
            transcript_path_to_read = master_transcript_path
            logging.info(f"Using first pass transcript: {transcript_path_to_read}")
        else:
            logging.warning(f"Master transcript not found in {result_dir}")

        if transcript_path_to_read:
            try:
                with open(transcript_path_to_read, 'r', encoding='utf-8') as f:
                    transcript = f.read()
            except Exception as read_err:
                logging.error(f"Error reading transcript file {transcript_path_to_read}: {read_err}")
                transcript = f"[Error reading transcript: {read_err}]"

        # --- Construct the expected zip file path ---
        # The zip file is saved in the parent directory of result_dir
        result_zip_file = None # Initialize to None
        try:
            # Extract the base name used for the zip from the result_dir name
            # Assumes result_dir is like '.../output/INPUT_BASENAME_TIMESTAMP'
            result_dir_name = os.path.basename(result_dir) # e.g., "INPUT_BASENAME_TIMESTAMP"
            # Find the last timestamp pattern (e.g., _YYYYMMDD_HHMMSS)
            timestamp_pattern = r"_(\d{8}_\d{6})$"
            match = re.search(timestamp_pattern, result_dir_name)
            if match:
                original_input_basename = result_dir_name[:match.start()] # Get part before timestamp
                
                parent_dir = os.path.dirname(result_dir) # e.g., "./whisper_output"
                # Reconstruct the zip filename pattern used in utils.zip_results
                # zip_filename = os.path.join(parent_dir, f"{base_name}_results_{os.path.basename(output_dir)}.zip")
                expected_zip_filename = f"{original_input_basename}_results_{result_dir_name}.zip"
                expected_zip_path = os.path.join(parent_dir, expected_zip_filename)

                if os.path.exists(expected_zip_path):
                    result_zip_file = expected_zip_path
                    logging.info(f"Found result zip file: {result_zip_file}")
                else:
                    logging.warning(f"Expected zip file not found at: {expected_zip_path}")
            else:
                logging.warning(f"Could not extract base name and timestamp from result directory: {result_dir_name}. Cannot reliably locate zip file.")
        
        except Exception as zip_find_err:
            logging.error(f"Error constructing or finding zip file path: {zip_find_err}")
        # --- End zip file path construction ---

        results_message = f"Processing complete! Results saved to {result_dir}"
        
        # Copy result file to temp dir for Gradio access
        final_result_path_for_gradio = None
        if result_zip_file:
            results_message += f"\nZip file created: {os.path.basename(result_zip_file)}"
            try:
                # Create a unique temp dir for this request
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, os.path.basename(result_zip_file))
                shutil.copy2(result_zip_file, temp_file_path)
                final_result_path_for_gradio = temp_file_path
                logging.info(f"Copied zip to temp location for download: {final_result_path_for_gradio}")
                 # Consider cleaning up older temp dirs if they accumulate
            except Exception as copy_err:
                logging.error(f"Error copying result zip to temp dir: {copy_err}")
                results_message += f"\nError preparing zip for download: {copy_err}"
                final_result_path_for_gradio = None # Ensure it's None if copy fails
        else:
             logging.warning("No result zip file found.")
             results_message += "\nResult zip file not found."
            
        return results_message, final_result_path_for_gradio, transcript
    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}")
        logging.error(traceback.format_exc())
        return f"An error occurred: {str(e)}\n\n{traceback.format_exc()}", None, ""

# Gradio interface
def build_interface():
    with gr.Blocks(title="WhisperBite - Audio Processing Tool") as demo:
        gr.Markdown("# 🎙️ WhisperBite")
        gr.Markdown("""
        This tool processes audio files by:
        1. Normalizing audio levels
        2. Separating speakers (diarization)
        3. Transcribing each speaker's audio
        4. Creating individual soundbites with transcripts
        """)

        with gr.Tabs():
            with gr.TabItem("Input"):
                with gr.Row():
                    with gr.Column():
                        # Add common video extensions
                        input_file = gr.File(
                            label="Input Audio or Video File", 
                            file_types=[
                                ".wav", ".mp3", ".m4a", ".ogg", ".flac", # Audio
                                ".mp4", ".mov", ".avi", ".mkv", ".webm" # Video
                            ]
                        )
                    with gr.Column():
                        input_folder = gr.Textbox(label="Input Folder Path", placeholder="Path to folder containing audio/video files", info="Processes the newest audio/video file in the folder.")
                    with gr.Column():
                        url = gr.Textbox(label="Audio URL", placeholder="YouTube or direct audio URL")
                
                output_folder = gr.Textbox(
                    label="Output Folder", 
                    placeholder="Path to save results",
                    value="./whisper_output"
                )
                
            with gr.TabItem("Processing Options"):
                with gr.Row():
                    model = gr.Dropdown(
                        label="Whisper Model", 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
                        value="turbo", # Default set to turbo
                        info="Larger models are more accurate but slower"
                    )
                    
                    num_speakers = gr.Slider(
                        label="Number of Speakers", 
                        minimum=1, 
                        maximum=10, 
                        step=1, 
                        value=2,
                        info="Set expected number of speakers. Diarization uses Pyannote (requires HF token)."
                    )
                
                with gr.Row():
                    auto_speakers = gr.Checkbox(
                        label="Auto-detect Speaker Count", 
                        value=False,
                        info="Automatically determine optimal speaker count (uses Pyannote, requires HF token)."
                    )
                    
                    enable_vocal_separation = gr.Checkbox(
                        label="Enable Vocal Separation", 
                        value=False,
                        info="Isolate voices from background noise/music (requires Demucs)"
                    )
                
                with gr.Row():
                    split_stereo = gr.Checkbox(
                        label="Split Stereo Channels (if stereo input)",
                        value=False,
                        info="Process Left and Right channels separately (useful for dual-mono recordings)"
                    )
                    force_mono_output = gr.Checkbox(
                        label="Force Mono Output Snippets",
                        value=False,
                        info="Convert all output speaker/word audio files to mono."
                    )
                
                with gr.Row():
                    enable_word_extraction = gr.Checkbox(
                        label="Enable Word Audio Extraction", 
                        value=False,
                        info="Extract individual word audio snippets (generates many files)"
                    )
                    
                    enable_second_pass = gr.Checkbox(
                        label="Enable Second Pass Refinement", 
                        value=False,
                        info="Perform extra analysis to refine speaker separation (experimental)"
                    )

                with gr.Row():
                    second_pass_min_duration = gr.Slider(
                        label="Second Pass Min Duration (s)",
                        minimum=0.5,
                        maximum=30.0,
                        step=0.5,
                        value=5.0,
                        info="Minimum segment length to consider for second pass refinement",
                        interactive=True # Start enabled, update based on enable_second_pass
                    )
                    
                    # Sound detection checkbox - interactivity handled later
                    attempt_sound_detection = gr.Checkbox(
                        label="Attempt Sound Detection (CLAP)",
                        value=False,
                        info="Identify non-speech sounds (requires Vocal Separation)",
                        interactive=False # Initially disabled
                    )

                # Add CLAP configuration sliders
                with gr.Row():
                    clap_chunk_duration = gr.Slider(
                        label="CLAP Chunk Duration (s)",
                        minimum=1.0,
                        maximum=10.0, # Adjust max as needed
                        step=0.5,
                        value=5.0,
                        info="Processing chunk size for CLAP sound detection",
                        interactive=False # Start disabled
                    )
                    clap_threshold = gr.Slider(
                        label="CLAP Detection Threshold",
                        minimum=0.1, 
                        maximum=1.0,
                        step=0.05,
                        value=0.7,
                        info="Confidence threshold for CLAP (higher = stricter)",
                        interactive=False # Start disabled
                    )

                # Add Textbox for CLAP prompts
                with gr.Row():
                    clap_target_prompts = gr.Textbox(
                        label="CLAP Target Prompts (Optional)",
                        placeholder="e.g., telephone ringing, dial tone, music",
                        info="Comma-separated list. If empty, uses defaults.",
                        lines=1,
                        interactive=False # Start disabled
                    )

                hf_token = gr.Textbox(label="Hugging Face Token", type="password", info="Required for speaker diarization. Get token from huggingface.co/settings/tokens.")
        
        submit_button = gr.Button("Process Audio", variant="primary")
        
        with gr.Row():
            output_message = gr.Textbox(label="Status", interactive=False, lines=3) # Increased lines for better messages
            result_file = gr.File(label="Download Results")
            transcript_preview = gr.TextArea(label="Transcript Preview", interactive=False, lines=10)

        submit_button.click(
            fn=run_pipeline,
            inputs=[
                input_file, input_folder, url, output_folder, model, 
                num_speakers, auto_speakers, enable_vocal_separation,
                enable_word_extraction, enable_second_pass,
                second_pass_min_duration,
                attempt_sound_detection,
                hf_token,
                split_stereo,
                clap_chunk_duration,
                clap_threshold,
                clap_target_prompts,
                force_mono_output
            ],
            outputs=[output_message, result_file, transcript_preview]
        )
        
        # Make Sound Detection checkbox interactive only if Vocal Separation is checked
        def update_sound_detection_interactivity(vocal_sep_enabled):
            # Also resets sound detection value if vocal sep is disabled
            # Use gr.update() for correct partial updates
            if vocal_sep_enabled:
                return gr.update(interactive=True)
            else:
                return gr.update(interactive=False, value=False)

        # Make CLAP sliders interactive only if Sound Detection is checked
        # Also handle the prompt textbox
        def update_clap_config_interactivity(sound_detect_enabled):
            updated_config = {
                clap_chunk_duration: gr.Slider(interactive=sound_detect_enabled),
                clap_threshold: gr.Slider(interactive=sound_detect_enabled),
                clap_target_prompts: gr.Textbox(interactive=sound_detect_enabled)
            }
            # If disabling sound detection, clear the prompts textbox
            if not sound_detect_enabled:
                 updated_config[clap_target_prompts] = gr.Textbox(value="", interactive=False)
                 
            return updated_config

        def update_second_pass_slider_interactivity(second_pass_enabled):
            return gr.Slider(interactive=second_pass_enabled)

        # Link vocal separation checkbox to sound detection interactivity
        enable_vocal_separation.change(
            fn=update_sound_detection_interactivity,
            inputs=enable_vocal_separation,
            outputs=attempt_sound_detection
        )

        # Link sound detection checkbox to CLAP config interactivity
        attempt_sound_detection.change(
            fn=update_clap_config_interactivity,
            inputs=attempt_sound_detection,
            outputs=[clap_chunk_duration, clap_threshold, clap_target_prompts] # Update outputs list
        )

        # Link second pass checkbox to slider interactivity
        enable_second_pass.change(
            fn=update_second_pass_slider_interactivity,
            inputs=enable_second_pass,
            outputs=second_pass_min_duration
        )

        # Add sample URLs only since file examples cause issues
        gr.Markdown("""
        ### Sample URLs to try:
        - Short video: https://www.youtube.com/watch?v=jNQXAC9IVRw
        - Interview example: https://www.youtube.com/watch?v=8S0FDjFBj8o
        """)
        
        # Examples with file paths don't work well with Gradio
        # Use sample configurations instead
        with gr.Accordion("Sample Configurations", open=False):
            gr.Markdown("""
            **Single Speaker Podcast:**
            - Model: base
            - Speakers: 1
            - Auto-detect: Disabled
            - Vocal Separation: Enabled
            
            **Interview Setup:**
            - Model: small
            - Speakers: 2
            - Auto-detect: Enabled
            - Vocal Separation: Enabled
            
            **Group Discussion:**
            - Model: medium
            - Speakers: 4
            - Auto-detect: Enabled
            - Vocal Separation: Disabled
            """)

    return demo

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WhisperBite - Audio Processing Tool")
    parser.add_argument("--public", action="store_true", help="Make the Gradio interface publicly accessible")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio interface on")
    
    args = parser.parse_args()
    
    interface = build_interface()
    interface.launch(share=args.public, server_port=args.port) # Removed allowed_paths
