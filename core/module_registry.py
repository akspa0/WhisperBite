class ModuleRegistry:
    def __init__(self):
        self._modules = {}

    def register(self, name, module):
        self._modules[name] = module

    def get(self, name):
        return self._modules.get(name)

    def list_modules(self):
        return list(self._modules.keys()) 