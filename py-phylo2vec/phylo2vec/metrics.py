"""Legacy alias for metrics module."""

import sys
import warnings

warnings.warn(
    "The 'phylo2vec.metrics' module is deprecated and will be removed in future versions. "
    "Please use 'phylo2vec.stats' instead.",
    DeprecationWarning,
)

sys.modules["metrics"] = sys.modules["phylo2vec.stats"]
