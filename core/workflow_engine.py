class WorkflowEngine:
    def __init__(self, module_registry):
        self.module_registry = module_registry

    def run(self, audio_obj, steps, config=None):
        for step in steps:
            module = self.module_registry.get(step)
            if module is None:
                raise ValueError(f"Module '{step}' not found in registry.")
            audio_obj = module(audio_obj, config=config or {})
        return audio_obj 