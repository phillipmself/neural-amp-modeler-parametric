"""
Parametric NAM models and datasets.

Importing this package installs the parametric registrations that are available in this
package.
"""

from ...data import register_dataset_initializer as _register_dataset_initializer
from ._dataset import data_config_from_model
from ._dataset import init_dataset as _init_dataset
from ._dataset import ParametricDataset
from ._hypernet import Hypernetwork
from ._spec import ParamSpec

_register_dataset_initializer("parametric", _init_dataset)

__all__ = ["Hypernetwork", "ParamSpec", "ParametricDataset", "data_config_from_model"]
