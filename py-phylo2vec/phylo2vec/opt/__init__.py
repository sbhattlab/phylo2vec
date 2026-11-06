"""Phylo2Vec-based optimisation methods.

GradME needs the optional dependencies: ``pip install phylo2vec[gradme]``.
"""

from ._api import list_methods
from ._bme_losses import bme_branch_lengths, bme_objective
from ._hc import HillClimbing

__all__ = ["HillClimbing", "bme_branch_lengths", "bme_objective", "list_methods"]

try:
    from ._gradme import GradME
    from ._gradme_losses import gradme_loss

    __all__ += ["GradME", "gradme_loss"]
except ImportError as err:
    # Without the [gradme] extras, the rest of phylo2vec.opt still imports.
    # Python calls a module's __getattr__ only for names it cannot find,
    # so asking for GradME then raises an error that says what to install.
    _GRADME_IMPORT_ERROR = err

    def __getattr__(name):
        if name in ("GradME", "gradme_loss"):
            raise ImportError(
                f"{name} needs the optional dependencies: pip install phylo2vec[gradme]"
            ) from _GRADME_IMPORT_ERROR
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
