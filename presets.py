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
    Implements a two-pass workflow.
    """
    # Define the workflow flags specifically for this preset
    workflow_config = {
        "detect_events": True,             # Enable Pass 1 CLAP
        "cut_between_events": True,      # NEW: Cut audio based on Pass 1 event boundaries
        "annotate_segments": True,       # NEW: Run Pass 2 CLAP and soundbite extraction on segments
        "transcribe": True,              # Enable transcription (will happen per-segment)
        "separate_vocals": False,        # Default disabled for this workflow
        "use_events_for_transcription": False, # Transcription guided by Pass 2 'speech' detection within segments
        "detect_sounds": False,          # Explicitly False, handled by annotate_segments
        "cut_segments": False,           # Explicitly False, replaced by cut_between_events
    }

    return {
        "name": "Event-Guided",
        "description": "Prioritizes detecting specific events (Pass 1) to segment audio, then detects sounds/speech within segments (Pass 2) and transcribes.",
        # <<< Wrap settings under 'config' key >>>
        "config": {
            "workflow": workflow_config,
            "event_detection": { # Pass 1 Config (for cutting)
                "target_events": kwargs.get("event_target_prompts", ["ringing phone", "hang-up tones"]),
                "threshold": kwargs.get("event_threshold", 0.5),
                "chunk_duration_s": kwargs.get("event_chunk_duration_s", 5.0),
                "min_duration": kwargs.get("event_min_duration", 1.0) # Corresponds to min_gap
            },
            "sound_detection": { # Pass 2 Config (for annotation within segments)
                "target_prompts": kwargs.get("sound_target_prompts", ["speech", "music", "sound effects", "laughter", "typing", "beep", "static", "background noise"]),
                "threshold": kwargs.get("sound_threshold", 0.3),
                "chunk_duration_s": kwargs.get("sound_chunk_duration_s", 5.0),
                "min_duration": kwargs.get("sound_min_duration", 1.0) # Pass 2 NMS gap
            },
            "transcription": { # Settings for Whisper transcription run on each segment
                "model_size": kwargs.get("transcription_model_size", "medium"),
                "word_timestamps": kwargs.get("transcription_word_timestamps", False)
            },
            # "diarization": { ... } # Placeholder for future
            "vocal_separation": {}, # Empty, as workflow disables it by default
            "segment_cutting": {} # Empty, as workflow uses cut_between_events
        }
    }

def get_transcribe_only_preset(**kwargs) -> Dict[str, Any]:
    """Transcription-only workflow preset."""
    workflow_config = _get_default_workflow_config() # Start with all False
    workflow_config["transcribe"] = True

    return {
        "name": "transcribe_only",
        "description": "Transcription only workflow",
        # <<< Ensure 'config' wrapper is present >>>
        "config": {
            "workflow": workflow_config,
            # Provide default empty configs for steps not used by this preset's workflow
            "transcription": {
                 "model": kwargs.get("model", "medium"), # Example: allow model override
                 "language": kwargs.get("language", None),
                 "task": kwargs.get("task", "transcribe")
            },
            "event_detection": {},
            "sound_detection": {},
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