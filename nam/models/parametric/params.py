# File: params.py
# Created Date: Sunday July 17th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Handling parametric inputs
"""

import inspect
from dataclasses import dataclass, fields
from typing import Any

from ..._core import InitializableFromConfig


@dataclass
class Param(InitializableFromConfig):
    default_value: Any

    @classmethod
    def init_from_config(cls, config):
        param_class, kwargs = cls.parse_config(config)
        return param_class(**kwargs)

    @classmethod
    def parse_config(cls, config):
        for candidate_class in [
            _class
            for _class in globals().values()
            if inspect.isclass(_class) and _class is not Param and issubclass(_class, Param)
        ]:
            if candidate_class.typestr() == config["type"]:
                config = dict(config)
                config.pop("type")
                break
        else:
            raise ValueError(f"Unrecognized parameter type {config['type']}")
        return candidate_class, config

    @classmethod
    def typestr(cls) -> str:
        raise NotImplementedError()

    def to_json(self):
        return {
            "type": self.typestr(),
            **{f.name: getattr(self, f.name) for f in fields(self)},
        }


@dataclass
class BooleanParam(Param):
    @classmethod
    def typestr(cls) -> str:
        return "boolean"


@dataclass
class ContinuousParam(Param):
    minval: float
    maxval: float

    @classmethod
    def typestr(cls) -> str:
        return "continuous"
