"""
Parametric NAM models and datasets.

Importing this package registers the parametric model + dataset. We intend to trigger
that import from the future ``nam.train.parametric`` entrypoint so non-parametric paths
stay untouched.
"""

from ._spec import ParamSpec

__all__ = ["ParamSpec"]
