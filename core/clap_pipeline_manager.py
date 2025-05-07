import os
import logging
from core import AudioObject
from core.rules_engine import RulesEngine

class ClapPipelineManager:
    def __init__(self, module_registry):
        self.registry = module_registry
        self.logger = logging.getLogger(__name__)

    def run(self, input_path, config):
        # Always run normalization first
        audio_obj = AudioObject(input_path, metadata={'input_file': input_path})
        normalize = self.registry.get('normalize_audio')
        audio_obj = normalize(audio_obj, config.get('step_configs', {}).get('normalize_audio', {}))
        self.logger.info('Normalization complete.')
        # Always run Demucs after normalization
        demucs = self.registry.get('separate_vocals_with_demucs')
        demucs_result = demucs(audio_obj, config.get('step_configs', {}).get('separate_vocals_with_demucs', {}))
        self.logger.info('Demucs separation complete.')
        # Assume demucs_result returns a dict with 'vocal' and 'no_vocal' AudioObjects
        vocal_obj = demucs_result.get('vocal')
        no_vocal_obj = demucs_result.get('no_vocal')
        # Run CLAP on no_vocal for music/ringing/tones
        annotate = self.registry.get('annotate_with_clap')
        no_vocal_clap_cfg = config.get('step_configs', {}).get('annotate_with_clap', {}).copy()
        no_vocal_clap_cfg['target_events'] = [
            "telephone ringing", "music", "hang-up tones"
        ]
        no_vocal_obj = annotate(no_vocal_obj, no_vocal_clap_cfg)
        self.logger.info('CLAP annotation on no_vocal complete.')
        # Optionally, run CLAP on vocal for speech/conversation
        vocal_clap_cfg = config.get('step_configs', {}).get('annotate_with_clap', {}).copy()
        vocal_clap_cfg['target_events'] = [
            "speech", "conversation"
        ]
        vocal_obj = annotate(vocal_obj, vocal_clap_cfg)
        self.logger.info('CLAP annotation on vocal complete.')
        # Merge CLAP events for segmentation: use no_vocal for music/ringing/tones, vocal for speech/conversation
        merged_clap_events = {}
        for k, v in no_vocal_obj.metadata.get('clap_events', {}).items():
            merged_clap_events[k] = v
        for k, v in vocal_obj.metadata.get('clap_events', {}).items():
            if k in merged_clap_events:
                merged_clap_events[k].extend(v)
            else:
                merged_clap_events[k] = v
        # Attach merged events to normalized audio for segmentation
        audio_obj.metadata['clap_events'] = merged_clap_events
        # Use rules engine to determine next steps based on CLAP results
        rules = config.get('rules', {})
        rules_engine = RulesEngine(rules)
        steps = rules_engine.determine_steps(audio_obj)
        self.logger.info(f'Workflow steps determined by rules engine: {steps}')
        step_configs = config.get('step_configs', {})
        # Run the rest of the pipeline as determined by rules
        for step in steps:
            module = self.registry.get(step)
            if not module:
                self.logger.warning(f'Module {step} not found in registry, skipping.')
                continue
            step_cfg = step_configs.get(step, {})
            merged_cfg = {**config, **step_cfg}
            # For segmentation, use normalized audio with merged CLAP events
            if step == 'segment_audio':
                audio_obj = module(audio_obj, merged_cfg)
                self.logger.info(f'Step {step} complete.')
            # For soundbite extraction, use vocal track and VAD, merge CLAP annotations
            elif step == 'extract_soundbites':
                # Optionally, run VAD here (not implemented)
                # Merge CLAP annotations into vocal_obj.metadata
                vocal_obj.metadata['clap_events'] = vocal_obj.metadata.get('clap_events', {})
                audio_obj = module(vocal_obj, merged_cfg)
                self.logger.info(f'Step {step} complete (vocal track).')
            else:
                audio_obj = module(audio_obj, merged_cfg)
                self.logger.info(f'Step {step} complete.')
        # If segmentation is part of the steps, process segments
        segments = audio_obj.metadata.get('segments', [])
        segment_results = []
        if segments:
            for seg in segments:
                seg_path = seg['path']
                seg_obj = AudioObject(seg_path, metadata={'parent': input_path})
                for step in steps[steps.index('segment_audio')+1:]:
                    module = self.registry.get(step)
                    if not module:
                        self.logger.warning(f'Module {step} not found in registry, skipping.')
                        continue
                    step_cfg = step_configs.get(step, {})
                    merged_cfg = {**config, **step_cfg}
                    seg_obj = module(seg_obj, merged_cfg)
                    self.logger.info(f'Segment step {step} complete.')
                segment_results.append(seg_obj)
            self.logger.info(f"CLAP pipeline complete. Segments processed: {len(segment_results)}")
            return segment_results
        else:
            self.logger.info("CLAP pipeline complete. No segments processed.")
            return [audio_obj] 