"""
Preset configurations for WhisperBite audio processing.
"""
from typing import Dict, Any, List, Optional
# Import defaults from sound_detection
from sound_detection import TARGET_SOUND_PROMPTS, DEFAULT_CALL_CHUNK_DURATION, DEFAULT_CALL_THRESHOLD

# Default events to detect
DEFAULT_EVENTS = [
    "telephone ringing",
    "speech",
    "conversation",
    "silence",
    "background noise",
    "phone dial tone",
    "phone busy signal"
]

# Define all possible workflow steps used in process_audio
ALL_WORKFLOW_STEPS = [
    "detect_events",
    "separate_vocals",
    "transcribe",
    "detect_sounds",
    "cut_segments", # Based on original presets, represents cutting at detected sounds
    "use_events_for_transcription" # Sub-option for transcription
]

def _get_default_workflow_config() -> Dict[str, bool]:
    """Returns a workflow config dict with all steps defaulting to False."""
    return {step: False for step in ALL_WORKFLOW_STEPS}

def get_standard_preset(**kwargs) -> Dict[str, Any]:
    """
    Standard preset with balanced settings for general use.
    Enables basic transcription and sound detection.
    """
    workflow_config = _get_default_workflow_config()
    workflow_config["transcribe"] = True
    workflow_config["detect_sounds"] = True
    # Ensure cutting happens if sounds are detected
    workflow_config["cut_segments"] = True if workflow_config["detect_sounds"] else False

    return {
        "name": "Standard",
        "description": "Balanced settings for general use",
        # Workflow config moved inside "config"
        "config": {
            "workflow": workflow_config,
            # Step-specific configurations remain here
            "transcription": {
                "model": kwargs.get("model", "base"),
                "language": kwargs.get("language", None),
                "task": kwargs.get("task", "transcribe")
            },
            "sound_detection": {
                "threshold": kwargs.get("clap_threshold", DEFAULT_CALL_THRESHOLD),
                "target_prompts": kwargs.get("clap_target_prompts", TARGET_SOUND_PROMPTS),
                "chunk_duration_s": kwargs.get("clap_chunk_duration", DEFAULT_CALL_CHUNK_DURATION)
            },
            "event_detection": {
                "threshold": kwargs.get("event_threshold", 0.5),
                "min_duration": kwargs.get("event_min_gap", 1.0),
                "target_events": kwargs.get("event_target_prompts", DEFAULT_EVENTS)
            },
            "vocal_separation": {},
            "segment_cutting": {}
        }
    }

def get_transcription_preset(**kwargs) -> Dict[str, Any]:
    """
    Transcription-focused preset optimized for speech.
    Enables advanced transcription features.
    """
    workflow_config = _get_default_workflow_config()
    workflow_config["separate_vocals"] = True
    workflow_config["transcribe"] = True

    return {
        "name": "Transcription",
        "description": "Optimized for speech transcription",
        # Workflow config moved inside "config"
        "config": {
            "workflow": workflow_config,
            # Step-specific configurations remain here
            "vocal_separation": {
                "model": kwargs.get("separation_model", "htdemucs"),
                "output_format": "wav"
            },
            "transcription": {
                "model": kwargs.get("model", "large-v3"),
                "language": kwargs.get("language", None),
                "task": kwargs.get("task", "transcribe"),
                "word_timestamps": True,
                "vad_filter": True
            },
            "event_detection": {
                "threshold": kwargs.get("event_threshold", 0.5),
                "min_duration": kwargs.get("event_min_gap", 1.0),
                "target_events": kwargs.get("event_target_prompts", DEFAULT_EVENTS)
            },
            "sound_detection": {
                "threshold": kwargs.get("clap_threshold", DEFAULT_CALL_THRESHOLD),
                "target_prompts": kwargs.get("clap_target_prompts", TARGET_SOUND_PROMPTS),
                "chunk_duration_s": kwargs.get("clap_chunk_duration", DEFAULT_CALL_CHUNK_DURATION)
            },
            "segment_cutting": {}
        }
    }

def get_event_guided_preset(**kwargs) -> Dict[str, Any]:
    """
    Event-guided preset for processing audio based on detected events.
    Enables event detection, vocal separation, and targeted transcription.
    """
    workflow_config = _get_default_workflow_config()
    workflow_config["detect_events"] = True
    workflow_config["separate_vocals"] = True
    workflow_config["transcribe"] = True
    workflow_config["detect_sounds"] = True
    workflow_config["cut_segments"] = True # Enable cutting based on detected sounds
    workflow_config["use_events_for_transcription"] = True # Use detected events for transcription

    return {
        "name": "Event-Guided",
        "description": "Process audio based on detected events",
        # Workflow config moved inside "config"
        "config": {
            "workflow": workflow_config,
            # Step-specific configurations remain here
            "event_detection": {
                "threshold": kwargs.get("event_threshold", 0.5),
                "min_duration": kwargs.get("event_min_gap", 1.0),
                "target_events": kwargs.get("event_target_prompts", DEFAULT_EVENTS)
            },
            "vocal_separation": {
                "model": kwargs.get("separation_model", "htdemucs"),
                "output_format": "wav"
            },
            "transcription": {
                "model": kwargs.get("model", "large-v3"),
                "language": kwargs.get("language", None),
                "task": kwargs.get("task", "transcribe"),
                "word_timestamps": True
            },
            "sound_detection": {
                "threshold": kwargs.get("clap_threshold", DEFAULT_CALL_THRESHOLD),
                "target_prompts": kwargs.get("clap_target_prompts", TARGET_SOUND_PROMPTS),
                "chunk_duration_s": kwargs.get("clap_chunk_duration", DEFAULT_CALL_CHUNK_DURATION)
            },
            "segment_cutting": {
                "min_segment_duration": kwargs.get("min_segment_duration", 1.0),
                "max_segment_duration": kwargs.get("max_segment_duration", 30.0),
                "overlap": kwargs.get("overlap", 0.0)
            }
        }
    }

def get_transcribe_only_preset(**kwargs) -> Dict[str, Any]:
    """Transcription-only workflow preset."""
    workflow_config = _get_default_workflow_config()
    workflow_config["transcribe"] = True

    return {
        "name": "transcribe_only",
        "description": "Transcription only workflow",
        # Workflow config moved inside "config"
        "config": {
            "workflow": workflow_config,
            "transcription": {},
            "event_detection": {
                "threshold": kwargs.get("event_threshold", 0.5),
                "min_duration": kwargs.get("event_min_gap", 1.0),
                "target_events": kwargs.get("event_target_prompts", DEFAULT_EVENTS)
            },
            "sound_detection": {
                "threshold": kwargs.get("clap_threshold", DEFAULT_CALL_THRESHOLD),
                "target_prompts": kwargs.get("clap_target_prompts", TARGET_SOUND_PROMPTS),
                "chunk_duration_s": kwargs.get("clap_chunk_duration", DEFAULT_CALL_CHUNK_DURATION)
            },
            "vocal_separation": {},
            "segment_cutting": {}
        }
    }

def get_available_presets() -> List[Dict[str, Any]]:
    """Get list of all available processing presets."""
    # Note: This will now return presets with the new structure
    # Ensure any code using this expects the new structure or adapt it.
    # The UI dropdown uses names, so it should be fine.
    return [
        get_standard_preset(),
        get_transcription_preset(), # Added Transcription preset to available list
        get_event_guided_preset(),
        get_transcribe_only_preset()
    ]

def get_preset_by_name(name: str) -> Dict[str, Any]:
    """Get preset configuration by name."""
    presets = {
        preset["name"]: preset
        for preset in get_available_presets()
    }
    
    if name not in presets:
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {list(presets.keys())}"
        )
    
    return presets[name] 