_model_registry = {}
def register_model(name=None):
    """
    wrapper to register model
    """
    def decorator(cls):
        key = name or cls.__name__
        if key in _model_registry:
            raise ValueError(f"Model {key} already registered.")
        _model_registry[key] = cls
        return cls
    return decorator

def get_model(name, *args, **kwargs):
    """
    get the model from the registery dict
    """
    if name not in _model_registry:
        raise KeyError(f"Model '{name}' not found in registry.")
    return _model_registry[name](*args, **kwargs)