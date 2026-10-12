"""离线难负样本挖掘；训练依赖仅在显式运行微调时加载。"""

from importlib import import_module

__all__ = ["HardNegativeMiner", "MiningConfig", "MiningReport", "export_training"]


def __getattr__(name: str):
    if name in __all__:
        value = getattr(import_module(".mining", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
