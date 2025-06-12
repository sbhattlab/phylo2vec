"""
API for registering and listing optimisation schemes.

Inspired by the `torchvision.models` API for registering models.
"""

from typing import List, Type

from phylo2vec.opt._base import BaseOptimizer

OPTIMIZER_REGISTRY = {}


def register_method(name: str = None) -> Type[BaseOptimizer]:
    """Decorator to register an optimisation scheme.

    Parameters
    ----------
    name : str, optional
        Name of the optimizer, by default None

    Returns
    -------
    Type[BaseOptimizer]
        Decorated class that inherits from BaseOptimizer.
    """

    def wrapper(cls: Type[BaseOptimizer]) -> Type[BaseOptimizer]:
        key = name if name else cls.__name__
        if key in OPTIMIZER_REGISTRY:
            raise ValueError(f"Optimization scheme '{key}' already registered.")
        OPTIMIZER_REGISTRY[key] = cls
        return cls

    return wrapper


def list_methods() -> List:
    """List all registered optimization schemes."""
    return list(OPTIMIZER_REGISTRY.keys())
