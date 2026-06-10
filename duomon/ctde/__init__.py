from __future__ import annotations

import importlib

_EXPORT_MODULES = (
    "ctde_features",
    "ctde_learning_context",
    "ctde_models",
    "ctde_data",
    "ctde_eval",
    "ctde_train_value",
    "ctde_train_pairwise",
)

__all__: list[str] = []

for _module_name in _EXPORT_MODULES:
    _module = importlib.import_module(f"{__name__}.{_module_name}")
    globals()[_module_name] = _module
    __all__.append(_module_name)
    for _name in getattr(_module, "__all__", ()):
        globals()[_name] = getattr(_module, _name)
        if _name not in __all__:
            __all__.append(_name)

del importlib, _EXPORT_MODULES, _module, _module_name, _name
