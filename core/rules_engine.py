class RulesEngine:
    def __init__(self, rules):
        self.rules = rules

    def determine_steps(self, audio_obj):
        # Use CLAP events to determine workflow steps dynamically
        clap_events = audio_obj.metadata.get('clap_events', {})
        # Example: If 'speech' or 'conversation' detected, run full pipeline; if only 'ringing', skip diarization/transcription
        steps = []
        if not clap_events or all(len(v) == 0 for v in clap_events.values()):
            # No events detected: fallback to minimal or error workflow
            return self.rules.get('no_events_steps', [])
        # Example logic: can be replaced with more complex rule parsing
        if any(clap_events.get(evt) for evt in ['speech', 'conversation']):
            steps = self.rules.get('speech_steps', [])
        elif any(clap_events.get(evt) for evt in ['telephone ringing', 'hang-up tones']):
            steps = self.rules.get('ringing_steps', [])
        else:
            steps = self.rules.get('default_steps', [])
        return steps

# Example rules YAML (to be placed in workflow config):
# rules:
#   speech_steps:
#     - segment_audio
#     - diarize_speakers
#     - transcribe_with_whisper
#     - extract_soundbites
#     - write_outputs
#   ringing_steps:
#     - segment_audio
#     - extract_soundbites
#     - write_outputs
#   no_events_steps:
#     - write_outputs
#   default_steps:
#     - segment_audio
#     - write_outputs 