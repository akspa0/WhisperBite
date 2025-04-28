import os
import logging
import zipfile
import subprocess
import shutil
from datetime import datetime
from pydub import AudioSegment
import yt_dlp
import json

def sanitize_filename(name, max_length=128):
    """Sanitize filename by removing unwanted characters and limiting length."""
    sanitized = "".join(char if char.isalnum() or char in " _-" else "_" for char in name)[:max_length]
    # Replace multiple consecutive underscores with a single one
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip()

def download_audio(url, output_dir, force_redownload=True):
    """Download audio from a URL and save it to the output directory with a unique filename."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s')
    }

    # Add a timestamp to ensure uniqueness if forcing redownload
    if force_redownload:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ydl_opts['outtmpl'] = os.path.join(output_dir, f'%(title)s_{timestamp}.%(ext)s')

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # The file will be saved with the .wav extension due to the postprocessor
            downloaded_file = os.path.join(output_dir, f"{info['title']}.wav")
            
            if not os.path.exists(downloaded_file):
                # If the postprocessor didn't work as expected, try to find the downloaded file
                for file in os.listdir(output_dir):
                    if info['title'] in file:
                        downloaded_file = os.path.join(output_dir, file)
                        break
                
            logging.info(f"Downloaded audio saved to {downloaded_file}")
            return downloaded_file
    except Exception as e:
        logging.error(f"Error downloading audio from {url}: {e}")
        raise

def get_media_info(file_path):
    """Extract media information using ffprobe."""
    if not file_path or not os.path.exists(file_path):
        logging.warning(f"Media info requested for non-existent file: {file_path}")
        return None

    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", file_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        media_info = json.loads(result.stdout)
        logging.info(f"Successfully extracted media info for {os.path.basename(file_path)}")
        return media_info
    except FileNotFoundError:
        logging.error("ffprobe command not found. Please ensure ffmpeg (which includes ffprobe) is installed and in your PATH.")
        return {"error": "ffprobe not found"}
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running ffprobe on {os.path.basename(file_path)}: {e}")
        logging.error(f"ffprobe stderr: {e.stderr}")
        return {"error": f"ffprobe failed: {e.stderr[:200]}..."}
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding ffprobe JSON output for {os.path.basename(file_path)}: {e}")
        return {"error": "Failed to decode ffprobe output"}
    except Exception as e:
        logging.error(f"Unexpected error getting media info for {os.path.basename(file_path)}: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def get_audio_channels(file_path):
    """Get the number of audio channels using pydub."""
    if not file_path or not os.path.exists(file_path):
        logging.warning(f"Channel count requested for non-existent file: {file_path}")
        return None
    try:
        audio = AudioSegment.from_file(file_path)
        channels = audio.channels
        logging.debug(f"Found {channels} channels for {os.path.basename(file_path)}")
        return channels
    except Exception as e:
        logging.error(f"Error getting channel count for {os.path.basename(file_path)}: {e}")
        # Don't return error dict here, None is fine for conditional checks
        return None 

def zip_results(output_dir, input_filename):
    """Zip the results directory contents into a single zip file."""
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    parent_dir = os.path.dirname(output_dir)
    zip_filename = os.path.join(parent_dir, f"{base_name}_results_{os.path.basename(output_dir)}.zip") 

    os.makedirs(parent_dir, exist_ok=True)

    logging.info(f"Creating zip archive: {zip_filename}")

    excluded_folders = ["normalized", "downloads", "stereo_split"] # Exclude stereo split temp dir
    excluded_files = [os.path.basename(zip_filename)] 

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            dirs[:] = [d for d in dirs if d not in excluded_folders]

            for file in files:
                if file in excluded_files:
                    continue
                # Ensure we include the YAML file and not the old TXT file
                if file == "master_transcript.txt":
                    continue 
                # Include the fallback JSON if YAML failed
                # if file == "master_transcript_fallback.json":
                #     pass # Allow fallback json
                # if file == "master_transcript_error.yaml":
                #     pass # Allow error yaml
                
                full_path = os.path.join(root, file)
                # Archive name should be relative to output_dir to maintain structure inside zip
                archive_name = os.path.relpath(full_path, output_dir)
                # Use forward slashes in archive name
                archive_name = archive_name.replace("\\", "/") 
                
                logging.debug(f"Adding to zip: {full_path} as {archive_name}")
                zipf.write(full_path, archive_name)

        # --- Metadata generation removed (now part of YAML) --- 

    logging.info(f"Successfully created zip file: {zip_filename}")
    return zip_filename

def create_thumbnail_waveform(audio_file, output_path):
    """Create a waveform thumbnail image for the audio file."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from pydub import AudioSegment
        
        # Load audio
        audio = AudioSegment.from_file(audio_file)
        samples = np.array(audio.get_array_of_samples())
        
        # Normalize
        samples = samples / np.max(np.abs(samples))
        
        # Create figure with transparent background
        fig, ax = plt.subplots(figsize=(5, 1))
        fig.patch.set_alpha(0)
        ax.set_alpha(0)
        
        # Plot waveform
        ax.plot(samples, color='blue', linewidth=0.5)
        ax.axis('off')
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, transparent=True, dpi=100)
        plt.close()
        
        return output_path
    except Exception as e:
        logging.warning(f"Error creating waveform thumbnail: {e}")
        return None
