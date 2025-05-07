class AudioObject:
    def __init__(self, path, metadata=None, provenance=None):
        self.path = path
        self.metadata = metadata or {}
        self.provenance = provenance or []

    def add_provenance(self, step, info=None):
        self.provenance.append({'step': step, 'info': info or {}}) 