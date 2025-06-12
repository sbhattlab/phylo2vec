"""
API for registering and listing optimisation schemes.

Inspired by the `torchvision.models` API for registering models.
"""

from typing import Callable, List

SCHEME_REGISTRY = {}


def register_model(name: str = None) -> Callable:
    def wrapper(cls: Callable) -> Callable:
        key = name if name else cls.__name__
        if key in SCHEME_REGISTRY:
            raise ValueError(f"Optimisation scheme '{key}' already registered.")
        SCHEME_REGISTRY[key] = cls
        return cls

    return wrapper


def list_models() -> List:
    """List all registered models."""
    return list(SCHEME_REGISTRY.keys())
