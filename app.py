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
import threading
import queue
import yaml
from typing import Dict, Any, Optional, List
from presets import (
    get_standard_preset,
    get_transcription_preset,
    get_event_guided_preset,
    get_preset_by_name
)
import datetime # For timestamp
from utils import sanitize_filename # For sanitizing filename

# Configure basic logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Global variables for job control
current_job: Optional[threading.Thread] = None
job_queue = queue.Queue()
stop_requested: bool = False
processing_stop_event: Optional[threading.Event] = None

def stop_current_job() -> None:
    """Request the current job to stop."""
    global stop_requested, processing_stop_event
    if current_job and current_job.is_alive():
        logger.info("Stop requested")
        stop_requested = True
        if processing_stop_event:
            processing_stop_event.set()
        return "Stop requested. The current job will terminate at the next safe point."
    return "No job is currently running."

def run_pipeline(
    input_file: str,
    output_dir: str,
    preset_name: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the audio processing pipeline with the selected preset.
    
    Args:
        input_file: Path to input audio/video file
        output_dir: Directory to save outputs
        preset_name: Name of the preset to use
        **kwargs: Additional arguments to pass to the preset
    
    Returns:
        Dict containing processing results and output paths
    """
    global current_job, stop_requested, processing_stop_event
    stop_requested = False
    processing_stop_event = threading.Event()
    file_handler = None # Initialize file handler variable
    
    try:
        # --- Create Unique Output Directory (Moved from process_wrapper) --- 
        # This ensures the log file path is determined before logging starts for the run
        base_output_folder = output_dir # Expecting unique dir path from wrapper now
        if not os.path.exists(base_output_folder):
            try:
                os.makedirs(base_output_folder, exist_ok=True)
                logger.info(f"Created output directory: {base_output_folder}")
            except OSError as e:
                 logger.error(f"Failed to create output directory {base_output_folder}: {e}")
                 # Return error early if dir creation fails
                 return {"status": "error", "error": f"Failed to create output dir: {e}"}
                 
        # --- Setup File Logging for this run --- 
        log_file_path = os.path.join(base_output_folder, "processing.log")
        file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite previous logs if run dir reused (shouldn't happen)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler) # Add handler to root logger
        logger.info(f"Logging for this run will be saved to: {log_file_path}")
        # --- End File Logging Setup ---
        
        # Get preset configuration
        preset_funcs = {
            "Standard": get_standard_preset,
            "Transcription": get_transcription_preset,
            "Event-Guided": get_event_guided_preset
        }
        
        if preset_name not in preset_funcs:
            raise ValueError(f"Unknown preset: {preset_name}")
            
        preset = preset_funcs[preset_name](**kwargs)
        logger.info(f"Using preset: {preset_name}")
        logger.debug(f"Preset config generated: {preset}")
        
        # Create output directory if it doesn't exist (redundant check, but safe)
        # os.makedirs(output_dir, exist_ok=True) # Output dir is now base_output_folder
        
        # Save preset configuration
        config_path = os.path.join(base_output_folder, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(preset, f)
        logger.info(f"Saved preset config to: {config_path}")
            
        # Process audio with preset configuration
        results = process_audio(
            input_file=input_file, 
            output_dir=base_output_folder, # Pass the unique run directory
            preset_name=preset_name, 
            preset_config=preset["config"], 
            stop_event=processing_stop_event
        )
        
        # Check if stopped after process_audio returned
        if stop_requested:
             logger.info("Pipeline stopped after process_audio completed.")
             # Decide how to handle this - maybe return a specific stopped status
             # For now, treat as success but maybe add a note
             results['status'] = 'stopped_post_processing'

        # Ensure results dictionary is returned even if process_audio had issues internally
        if not isinstance(results, dict):
             logger.error(f"process_audio returned unexpected type: {type(results)}")
             results = {"status": "error", "error": "Internal processing function error."}

        # Add paths relative to the run directory if possible
        # Example: (adjust keys based on actual results dict from process_audio)
        if results.get("status") not in ["error", "cancelled", "stopped_post_processing"]:
            results["output_run_directory"] = base_output_folder
            # Add other key paths if needed

        # Update status based on process_audio result before returning
        final_status = results.get("status", "unknown")

        return {
            "status": final_status, 
            "preset_used": preset_name,
            "config_file": config_path,
            **results
        }
        
    except Exception as e:
        logger.exception("Error in processing pipeline (run_pipeline)")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        # --- Remove File Handler for this run --- 
        if file_handler:
            logger.info(f"Removing file handler for {log_file_path}")
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
        # --- End File Handler Removal ---

def process_wrapper(
    input_file,
    input_folder,
    url,
    output_folder,
    model,
    num_speakers,
    auto_speakers,
    enable_vocal_separation,
    enable_word_extraction,
    enable_second_pass,
    second_pass_min_duration,
    hf_token,
    split_stereo,
    force_mono_output,
    preset,
    event_target_prompts_input,
    event_threshold,
    event_min_gap,
    clap_target_prompts_input,
    clap_threshold_input,
    clap_chunk_duration_input
):
    """Wrapper function to handle Gradio interface inputs."""
    global current_job, stop_requested, processing_stop_event
    stop_requested = False
    processing_stop_event = threading.Event() # Ensure event is reset/created for the job
    
    try:
        # Determine input file path
        input_source_description = ""
        if input_file is not None:
            file_path = input_file.name
            input_source_description = os.path.basename(file_path)
        elif input_folder:
            # Get newest file from folder
            files = [(os.path.getmtime(os.path.join(input_folder, f)), os.path.join(input_folder, f))
                    for f in os.listdir(input_folder)]
            if not files:
                return {"status": "error", "error": "No files found in input folder"}
            file_path = max(files)[1]
            input_source_description = os.path.basename(file_path)
        elif url:
            # Download from URL
            download_dir = os.path.join(output_folder, "downloads")
            os.makedirs(download_dir, exist_ok=True)
            file_path = download_audio(url, download_dir)
            if not file_path:
                return {"status": "error", "error": "Failed to download audio from URL"}
            # Try to get a reasonable name from URL or downloaded file
            input_source_description = os.path.basename(file_path) if file_path else url.split('/')[-1]
        else:
            return {"status": "error", "error": "Please provide an input file, folder, or URL"}

        # Create Unique Output Directory
        base_output_folder = output_folder # The folder selected in UI
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_input_name = sanitize_filename(os.path.splitext(input_source_description)[0])
        # Limit length of sanitized name to avoid overly long paths
        max_name_len = 50 
        unique_run_folder_name = f"{sanitized_input_name[:max_name_len]}_{timestamp}"
        run_output_dir = os.path.join(base_output_folder, unique_run_folder_name)
        
        try:
            os.makedirs(run_output_dir, exist_ok=True)
            logger.info(f"Created unique output directory: {run_output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {run_output_dir}: {e}")
            return f"Error creating output directory: {e}", None, None
            
        # Prepare kwargs for preset configuration
        preset_kwargs = {
            "model": model,
            "num_speakers": num_speakers,
            "auto_speakers": auto_speakers,
            "enable_vocal_separation": enable_vocal_separation,
            "enable_word_extraction": enable_word_extraction,
            "enable_second_pass": enable_second_pass,
            "second_pass_min_duration": second_pass_min_duration,
            "hf_token": hf_token,
            "split_stereo": split_stereo,
            "force_mono_output": force_mono_output,
            "event_target_prompts": [p.strip() for p in event_target_prompts_input.split(",") if p.strip()],
            "event_threshold": event_threshold,
            "event_min_gap": event_min_gap,
            "clap_target_prompts": [p.strip() for p in clap_target_prompts_input.split(",") if p.strip()],
            "clap_threshold": clap_threshold_input,
            "clap_chunk_duration": clap_chunk_duration_input
        }

        # Call run_pipeline with the unique run directory
        results = run_pipeline(
            input_file=file_path,
            output_dir=run_output_dir,
            preset_name=preset,
            **preset_kwargs
        )
        
        # Process the results for Gradio output
        if results.get("status") == "error":
            error_message = results.get("error", "Unknown processing error")
            return f"Error: {error_message}", None, None # Return 3 values on error
        elif results.get("status") == "cancelled":
            return "Processing cancelled.", None, None # Return 3 values on cancel
        elif results.get("status") == "stopped_post_processing":
            return "Processing stopped.", None, None # Return 3 values on stop
        else:
            # Success case
            output_zip = results.get("output_zip_path")
            master_transcript_path = results.get("master_transcript_path") # Assuming process_audio returns this
            preview_text = ""
            if master_transcript_path and os.path.exists(master_transcript_path):
                try:
                    with open(master_transcript_path, 'r') as f:
                        # Limit preview size
                        preview_text = f.read(2000) 
                        if len(preview_text) == 2000:
                            preview_text += "... (truncated)"
                except Exception as e:
                    logger.warning(f"Could not read transcript for preview: {e}")
            
            status_message = f"Processing complete. Preset: {results.get('preset_used', 'N/A')}"
            return status_message, output_zip, preview_text # Return 3 values on success

    except Exception as e:
        logger.exception("Error in process_wrapper")
        error_message = str(e)
        return f"Wrapper Error: {error_message}", None, None # Return 3 values on wrapper error

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
                preset = gr.Dropdown(
                    choices=["Standard", "Transcription", "Event-Guided"],
                    value="Standard",
                    label="Processing Preset",
                    info="Select a processing workflow. Detection options below depend on the chosen preset."
                )
                
                # Standard Options
                with gr.Group(visible=True) as standard_options_group:
                    gr.Markdown("### General Processing")
                    with gr.Row():
                        model = gr.Dropdown(
                            label="Whisper Model", 
                            choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
                            value="turbo", 
                            info="Larger models are more accurate but slower"
                        )
                        hf_token = gr.Textbox(label="Hugging Face Token", type="password", info="Required for some models/diarization. huggingface.co/settings/tokens.")
                    
                    gr.Markdown("### Speaker Diarization")
                    gr.HTML("<p style='font-size:small;color:grey'>Requires Pyannote & HF Token. Only runs if preset includes transcription/diarization.</p>") # Info text
                    with gr.Row():
                        num_speakers = gr.Slider(
                            label="Number of Speakers", minimum=1, maximum=10, step=1, value=2,
                            interactive=True # Assume enabled unless preset disables diarization
                        )
                        auto_speakers = gr.Checkbox(
                            label="Auto-detect Speaker Count", value=False,
                            interactive=True # Assume enabled unless preset disables diarization
                        )
                        
                    gr.Markdown("### Audio Handling")
                    with gr.Row():
                        enable_vocal_separation = gr.Checkbox(
                            label="Enable Vocal Separation (Demucs)", value=False,
                            interactive=True # Assume enabled unless preset disables it
                        )
                        split_stereo = gr.Checkbox(
                            label="Split Stereo Channels (if stereo input)", value=False,
                            info="Process L/R channels separately"
                        )
                        force_mono_output = gr.Checkbox(
                            label="Force Mono Output Snippets", value=False,
                            info="Convert all output speaker/word audio files to mono."
                        )
                        
                    gr.Markdown("### Advanced Features")
                    with gr.Row():
                        enable_word_extraction = gr.Checkbox(
                            label="Enable Word Audio Extraction", value=False,
                            info="Extract individual word audio snippets (generates many files)"
                        )
                        enable_second_pass = gr.Checkbox(
                            label="Enable Second Pass Diarization Refinement", value=False,
                            info="Perform extra analysis to refine speaker separation (experimental)"
                        )
                    with gr.Row():
                        second_pass_min_duration = gr.Slider(
                            label="Second Pass Min Duration (s)", minimum=0.5, maximum=30.0, step=0.5, value=5.0,
                            info="Min segment length for second pass",
                            interactive=False # Depends on enable_second_pass
                        )
                        
            # <<< NEW DETECTION TAB >>>
            with gr.TabItem("Detection"):
                gr.Markdown("Configure settings for **Event Detection** and **Sound Detection (CLAP)**. These steps only run if the selected Preset includes them.")
                
                with gr.Group() as event_detection_group: # Group for event detection
                    gr.Markdown("### Event Detection Settings")
                    gr.HTML("<p style='font-size:small;color:grey'>Detects broader audio categories (e.g., speech, music, noise). Output: `events/events.json`.</p>")
                    event_target_prompts_input = gr.Textbox(
                        label="Target Events (comma-separated)",
                        placeholder="e.g., speech, music, noise, telephone ringing",
                        info="Prompts for event detection. Leave empty to use defaults.",
                        lines=1,
                        interactive=False # Depends on preset
                    )
                    with gr.Row():
                        event_threshold = gr.Slider(
                            label="Event Detection Threshold", minimum=0.1, maximum=1.0, step=0.05, value=0.5, # Lower default maybe?
                            info="Confidence threshold for detecting events.",
                            interactive=False # Depends on preset
                        )
                        event_min_gap = gr.Slider(
                             label="Event Min Gap (s)", minimum=0.1, maximum=10.0, step=0.1, value=1.0,
                             info="Minimum time gap between detected events of the same type.",
                             interactive=False # Depends on preset
                        )
                        
                with gr.Group() as sound_detection_group: # Group for sound detection
                    gr.Markdown("### Sound Detection (CLAP) Settings")
                    gr.HTML("<p style='font-size:small;color:grey'>Detects specific sounds using CLAP. Output: `sounds/sounds.json` (or similar). Often requires Vocal Separation.</p>")
                    clap_target_prompts_input = gr.Textbox(
                        label="Target Sounds (comma-separated)", 
                        placeholder="e.g., telephone ringing, dial tone, laughter, dog barking", 
                        info="Prompts for CLAP sound detection. Leave empty to use defaults.", 
                        lines=1,
                        interactive=False # Depends on preset
                    )
                    with gr.Row():
                        clap_threshold_input = gr.Slider(
                            label="CLAP Detection Threshold", minimum=0.1, maximum=1.0, step=0.05, value=0.7,
                            info="Confidence threshold for CLAP (higher = stricter)",
                            interactive=False # Depends on preset
                        )
                        clap_chunk_duration_input = gr.Slider(
                            label="CLAP Chunk Duration (s)", minimum=1.0, maximum=10.0, step=0.5, value=5.0,
                            info="Processing chunk size for CLAP",
                            interactive=False # Depends on preset
                        )

        with gr.Row():
            submit_button = gr.Button("Process Audio", variant="primary")
            stop_button = gr.Button("Stop Processing", variant="stop")
        
        with gr.Row():
            output_message = gr.Textbox(label="Status", interactive=False, lines=3)
            result_file = gr.File(label="Download Results")
            transcript_preview = gr.TextArea(label="Transcript Preview", interactive=False, lines=10)

        # Connect the stop button
        stop_button.click(
            fn=stop_current_job,
            inputs=[],
            outputs=[output_message]
        )

        # --- Interactivity Logic --- 

        # Helper function (should be defined before use)
        def update_second_pass_slider_interactivity(second_pass_enabled):
            """Update second pass slider interactivity based on checkbox."""
            return gr.update(interactive=second_pass_enabled)

        # Link second pass checkbox to slider interactivity
        enable_second_pass.change(
            fn=update_second_pass_slider_interactivity,
            inputs=enable_second_pass,
            outputs=second_pass_min_duration
        )

        # NEW: Link Preset dropdown to Detection tab controls
        def update_detection_interactivity(preset_name):
            try:
                preset_data = get_preset_by_name(preset_name)
                workflow = preset_data.get("config", {}).get("workflow", {})
                
                event_detect_enabled = workflow.get("detect_events", False)
                sound_detect_enabled = workflow.get("detect_sounds", False)
                
                # Update Event Detection Group
                event_updates = [
                    gr.update(interactive=event_detect_enabled),
                    gr.update(interactive=event_detect_enabled),
                    gr.update(interactive=event_detect_enabled)
                ]
                
                # Update Sound Detection Group
                sound_updates = [
                    gr.update(interactive=sound_detect_enabled),
                    gr.update(interactive=sound_detect_enabled),
                    gr.update(interactive=sound_detect_enabled)
                ]
                
                # Combine updates for all controls in order
                # Order: event_target_prompts, event_threshold, event_min_gap, 
                #        clap_target_prompts, clap_threshold, clap_chunk_duration
                return tuple(event_updates + sound_updates)

            except Exception as e:
                logger.error(f"Error updating detection interactivity for preset '{preset_name}': {e}")
                # Return updates to disable everything on error
                return tuple([gr.update(interactive=False)] * 6) 

        preset.change(
            fn=update_detection_interactivity,
            inputs=preset,
            outputs=[
                # Event Detection Controls
                event_target_prompts_input, 
                event_threshold, 
                event_min_gap,
                # Sound Detection Controls
                clap_target_prompts_input,
                clap_threshold_input,
                clap_chunk_duration_input
            ]
        )
        
        # --- Submit Button Call --- 
        submit_button.click(
            fn=process_wrapper,
            inputs=[
                # Input Tab
                input_file, 
                input_folder, 
                url, 
                output_folder, 
                # Processing Options Tab
                model, 
                num_speakers, 
                auto_speakers, 
                enable_vocal_separation,
                enable_word_extraction, 
                enable_second_pass,
                second_pass_min_duration,
                hf_token,
                split_stereo,
                force_mono_output,
                preset,
                # Detection Tab
                event_target_prompts_input,
                event_threshold,
                event_min_gap,
                clap_target_prompts_input,
                clap_threshold_input,
                clap_chunk_duration_input
            ],
            outputs=[output_message, result_file, transcript_preview]
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
