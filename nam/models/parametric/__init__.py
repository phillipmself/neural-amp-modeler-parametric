"""
Parametric NAM models and datasets.

Importing this package registers the parametric model + dataset. We intend to trigger
that import from the future ``nam.train.parametric`` entrypoint so non-parametric paths
stay untouched.
"""

from ...data import register_dataset_initializer as _register_dataset_initializer
from ._dataset import data_config_from_model
from ._dataset import init_dataset as _init_dataset
from ._dataset import ParametricDataset
from ._spec import ParamSpec

_register_dataset_initializer("parametric", _init_dataset)

__all__ = ["ParamSpec", "ParametricDataset", "data_config_from_model"]
