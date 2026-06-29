"""
Parametric NAM models and datasets.

Importing this package installs the parametric registrations that are available in this
package.
"""

from ...data import register_dataset_initializer as _register_dataset_initializer
from .. import factory as _factory
from ._dataset import data_config_from_model
from ._dataset import init_dataset as _init_dataset
from ._dataset import ParametricDataset
from ._export import bake
from ._export import bake_to_files
from ._export import export_parametric
from ._export import output_scale_from_datasets
from ._concat_lstm import ConcatLSTM
from ._hypernet import Hypernetwork
from ._hyperwavenet import HyperWaveNet
from ._spec import ParamSpec

_register_dataset_initializer("parametric", _init_dataset)
_factory.register("ConcatLSTM", ConcatLSTM.init_from_config)
_factory.register("HyperWaveNet", HyperWaveNet.init_from_config)

__all__ = [
    "ConcatLSTM",
    "Hypernetwork",
    "HyperWaveNet",
    "ParamSpec",
    "ParametricDataset",
    "bake",
    "bake_to_files",
    "data_config_from_model",
    "export_parametric",
    "output_scale_from_datasets",
]
