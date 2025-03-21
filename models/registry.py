import mmcv
from mmcv.utils import Registry

# def _build_func(name: str, option: mmcv.ConfigDict, registry: Registry):
#     return registry.get(name)(option)

def _build_func(name: str, option, registry: Registry):
    model_class = registry.get(name)
    if model_class is None:
        raise ValueError(f"Model {name} not found in the registry.")
    return model_class(option)

MODELS = Registry('models', build_func=_build_func)
