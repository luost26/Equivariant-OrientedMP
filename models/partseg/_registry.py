
_MODEL_DICT = {}


def register_model(name):
    def decorator(cls):
        _MODEL_DICT[name] = cls
        return cls
    return decorator


def get_model(cfg, num_classes):
    return _MODEL_DICT[cfg.type](cfg, num_classes)
