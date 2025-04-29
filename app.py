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
    get_event_guided_preset
)

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
    
    try:
        # Get preset configuration
        preset_funcs = {
            "Standard": get_standard_preset,
            "Transcription": get_transcription_preset,
            "Event-Guided": get_event_guided_preset
        }
        
        if preset_name not in preset_funcs:
            raise ValueError(f"Unknown preset: {preset_name}")
            
        preset = preset_funcs[preset_name](**kwargs)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preset configuration
        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(preset, f)
            
        # Process audio with preset configuration
        results = process_audio(
            input_file=input_file,
            output_dir=output_dir,
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

        return {
            "status": "success",
            "preset_used": preset_name,
            "config_file": config_path,
            **results
        }
        
    except Exception as e:
        logger.exception("Error in processing pipeline")
        return {
            "status": "error",
            "error": str(e)
        }

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
    attempt_sound_detection,
    hf_token,
    split_stereo,
    clap_chunk_duration,
    clap_threshold,
    clap_target_prompts,
    force_mono_output,
    preset
):
    """Wrapper function to handle Gradio interface inputs."""
    try:
        # Determine input file path
        if input_file is not None:
            file_path = input_file.name
        elif input_folder:
            # Get newest file from folder
            files = [(os.path.getmtime(os.path.join(input_folder, f)), os.path.join(input_folder, f))
                    for f in os.listdir(input_folder)]
            if not files:
                return {"status": "error", "error": "No files found in input folder"}
            file_path = max(files)[1]
        elif url:
            # Download from URL
            download_dir = os.path.join(output_folder, "downloads")
            os.makedirs(download_dir, exist_ok=True)
            file_path = download_audio(url, download_dir)
            if not file_path:
                return {"status": "error", "error": "Failed to download audio from URL"}
        else:
            return {"status": "error", "error": "Please provide an input file, folder, or URL"}

        # Prepare kwargs for preset configuration
        preset_kwargs = {
            "model": model,
            "num_speakers": num_speakers,
            "auto_speakers": auto_speakers,
            "enable_vocal_separation": enable_vocal_separation,
            "enable_word_extraction": enable_word_extraction,
            "enable_second_pass": enable_second_pass,
            "second_pass_min_duration": second_pass_min_duration,
            "attempt_sound_detection": attempt_sound_detection,
            "hf_token": hf_token,
            "split_stereo": split_stereo,
            "clap_chunk_duration": clap_chunk_duration,
            "clap_threshold": clap_threshold,
            "clap_target_prompts": clap_target_prompts.split(",") if clap_target_prompts else None,
            "force_mono_output": force_mono_output
        }

        # Call run_pipeline with the correct arguments
        results = run_pipeline(
            input_file=file_path,
            output_dir=output_folder,
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
                # <<< Add Preset Radio Buttons >>>
                preset = gr.Dropdown(
                    choices=["Standard", "Transcription", "Event-Guided"],
                    value="Standard",
                    label="Processing Preset"
                )
                
                # Standard Options (potentially hide/show based on mode later)
                with gr.Group(visible=True) as standard_options_group:
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
                            info="Isolate voices from background noise/music (required for Call Analysis mode)."
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

                # CLAP Options (Visibility depends on vocal sep & maybe mode later)
                with gr.Group(visible=True) as clap_options_group: 
                    with gr.Row():
                        attempt_sound_detection = gr.Checkbox(
                            label="Enable Non-Vocal Sound Detection",
                            value=False,
                            info="Identify non-speech sounds in the non-vocal track (requires Vocal Separation)",
                            interactive=False # Still depends on vocal sep
                        )
                        enable_vocal_clap = gr.Checkbox(
                            label="Enable Vocal Sound Detection",
                            value=False,
                            info="Identify vocal cues like laughter in the vocal track (requires Vocal Separation)",
                            interactive=False # Also depends on vocal sep
                        )
                    with gr.Row():
                        clap_chunk_duration = gr.Slider(
                            label="CLAP Chunk Duration (s)",
                            minimum=1.0,
                            maximum=10.0, # Adjust max as needed
                            step=0.5,
                            value=5.0,
                            info="Processing chunk size for CLAP sound detection",
                            interactive=False # Depends on either CLAP option being enabled
                        )
                        clap_threshold = gr.Slider(
                            label="CLAP Detection Threshold",
                            minimum=0.1, 
                            maximum=1.0,
                            step=0.05,
                            value=0.7,
                            info="Confidence threshold for CLAP (higher = stricter)",
                            interactive=False # Depends on either CLAP option being enabled
                        )
                    with gr.Row():
                        clap_target_prompts = gr.Textbox(
                            label="CLAP Target Prompts (Optional)",
                            placeholder="e.g., telephone ringing, dial tone, music, laughter",
                            info="Comma-separated list. If empty, uses defaults based on enabled detection types.",
                            lines=1,
                            interactive=False # Depends on either CLAP option being enabled
                        )

                hf_token = gr.Textbox(label="Hugging Face Token", type="password", info="Required for speaker diarization (Standard mode) or Pyannote models. Get token from huggingface.co/settings/tokens.")
        
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

        submit_button.click(
            fn=process_wrapper,
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
                force_mono_output,
                preset
            ],
            outputs=[output_message, result_file, transcript_preview]
        )
        
        # Update interactivity functions
        def update_sound_detection_interactivity(vocal_sep_enabled):
            """Update both CLAP detection checkboxes based on vocal separation."""
            if vocal_sep_enabled:
                return [
                    gr.update(interactive=True),  # For non-vocal detection
                    gr.update(interactive=True)   # For vocal detection
                ]
            else:
                return [
                    gr.update(interactive=False, value=False),  # For non-vocal detection
                    gr.update(interactive=False, value=False)   # For vocal detection
                ]

        def update_clap_config_interactivity(non_vocal_enabled, vocal_enabled):
            """Update CLAP configuration controls based on either detection being enabled."""
            any_clap_enabled = non_vocal_enabled or vocal_enabled
            return [
                gr.update(interactive=any_clap_enabled),  # chunk duration
                gr.update(interactive=any_clap_enabled),  # threshold
                gr.update(interactive=any_clap_enabled,   # prompts
                         value="" if not any_clap_enabled else None)  # Clear if disabled
            ]

        def update_second_pass_slider_interactivity(second_pass_enabled):
            """Update second pass slider interactivity based on checkbox."""
            return gr.update(interactive=second_pass_enabled)

        # Link vocal separation checkbox to both CLAP detection interactivity
        enable_vocal_separation.change(
            fn=update_sound_detection_interactivity,
            inputs=enable_vocal_separation,
            outputs=[attempt_sound_detection, enable_vocal_clap]
        )

        # Link both CLAP detection checkboxes to config interactivity
        for clap_checkbox in [attempt_sound_detection, enable_vocal_clap]:
            clap_checkbox.change(
                fn=update_clap_config_interactivity,
                inputs=[attempt_sound_detection, enable_vocal_clap],
                outputs=[clap_chunk_duration, clap_threshold, clap_target_prompts]
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
