import yaml
import logging
import os
import datetime
from core import AudioObject, ModuleRegistry, WorkflowEngine
from core.clap_pipeline_manager import ClapPipelineManager
from modules import (
    normalize_audio, annotate_with_clap, segment_audio, separate_vocals_with_demucs,
    diarize_speakers, transcribe_with_whisper, extract_soundbites, write_outputs
)
from config.presets import CANONICAL_WORKFLOW

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WhisperBite Modular CLAP Pipeline")
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--hf_token", required=False, help="Hugging Face token for diarization")
    parser.add_argument("--model", default="base", help="Whisper model name")
    parser.add_argument("--enable_vocal_separation", action="store_true", help="Enable Demucs vocal separation")
    parser.add_argument("--enable_word_extraction", action="store_true", help="Enable word-level timestamp extraction")
    parser.add_argument("--workflow", default="workflows/canonical_clap_workflow.yaml", help="Path to workflow YAML file")
    args = parser.parse_args()

    # Create timestamped run directory
    input_base = os.path.splitext(os.path.basename(args.input))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{input_base}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    # Create subfolders for each pipeline stage
    subfolders = ["normalized", "clap", "segments", "diarization", "transcription", "soundbites", "logs"]
    for sub in subfolders:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    log_path = os.path.join(run_dir, "logs", "processing.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode='w')
        ]
    )
    logger = logging.getLogger("WhisperBite")

    # Load workflow YAML
    with open(args.workflow, 'r') as f:
        workflow = yaml.safe_load(f)
    steps = workflow.get('steps', [])
    step_configs = workflow.get('config', {})
    rules = workflow.get('rules', {})

    # Build global config from args
    config = {
        'output_dir': run_dir,
        'hf_token': args.hf_token,
        'model': args.model,
        'enable_vocal_separation': args.enable_vocal_separation,
        'enable_word_extraction': args.enable_word_extraction,
        'workflow_steps': steps,
        'step_configs': step_configs,
        'rules': rules,
        'run_dir': run_dir,
        'stage_dirs': {sub: os.path.join(run_dir, sub) for sub in subfolders},
    }

    # Initialize module registry and register all modules
    registry = ModuleRegistry()
    registry.register('normalize_audio', normalize_audio)
    registry.register('annotate_with_clap', annotate_with_clap)
    registry.register('segment_audio', segment_audio)
    registry.register('separate_vocals_with_demucs', separate_vocals_with_demucs)
    registry.register('diarize_speakers', diarize_speakers)
    registry.register('transcribe_with_whisper', transcribe_with_whisper)
    registry.register('extract_soundbites', extract_soundbites)
    registry.register('write_outputs', write_outputs)

    print(f"[WhisperBite] Starting pipeline in run directory: {run_dir}")
    logger.info(f"Pipeline started in run directory: {run_dir}")
    pipeline_manager = ClapPipelineManager(registry)
    results = pipeline_manager.run(args.input, config)
    print("[WhisperBite] Pipeline complete.")
    logger.info("Pipeline complete.")

    # Print summary
    if results and hasattr(results[0], 'metadata'):
        audio_obj = results[0]
        clap_events = audio_obj.metadata.get('clap_events', {})
        segments = audio_obj.metadata.get('segments', [])
        print("[WhisperBite] CLAP events detected:")
        logger.info(f"CLAP events detected: { {k: len(v) for k, v in clap_events.items()} }")
        for k, v in clap_events.items():
            print(f"  {k}: {len(v)} event(s)")
        print(f"[WhisperBite] Workflow path: {config.get('rules', {})}")
        print(f"[WhisperBite] Segments produced: {len(segments) if segments else 0}")
        if segments:
            for seg in segments:
                print(f"  Segment: {seg.get('path', 'unknown')}")
        print(f"[WhisperBite] Normalized WAV: {audio_obj.path}")
    else:
        print("[WhisperBite] No results or metadata available.")
        logger.warning("No results or metadata available.")
    print(f"[WhisperBite] Log file: {log_path}")
    print("[WhisperBite] Output directory structure:")
    for sub in subfolders:
        print(f"  {sub}/ -> {os.path.join(run_dir, sub)}")
