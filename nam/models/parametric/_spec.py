"""
ParamSpec: a self-describing parameter specification for parametric NAM models.

Each ParamSpec captures everything the model and downstream tooling need to know
about one continuous control parameter: its name (for UI display), its numeric
range (for plugin/UI clamping), and its default value (for export snapshots and
loudness normalization).

Design note: min/max are metadata for downstream consumers (plugins, UIs) and
do NOT affect the forward pass, normalization, or training math. The net consumes
default values positionally at export time, exactly as the old nominal_params did.
Order is significant — the list position is the positional index into the params
tensor.
"""

from dataclasses import dataclass
import math


@dataclass
class ParamSpec:
    """Specification for one continuous control parameter.

    Attributes
    ----------
    name:    Human-readable identifier, e.g. ``"gain"`` or ``"bright"``.
    min:     Minimum valid value for this parameter (inclusive).
    max:     Maximum valid value for this parameter (inclusive).
    default: Default (nominal) value, used for export snapshots.

    Constraints (enforced at construction):
    - All values must be finite.
    - min <= default <= max.
    """

    name: str
    min: float
    max: float
    default: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.min):
            raise ValueError(
                f"ParamSpec '{self.name}': min={self.min} is not finite."
            )
        if not math.isfinite(self.max):
            raise ValueError(
                f"ParamSpec '{self.name}': max={self.max} is not finite."
            )
        if not math.isfinite(self.default):
            raise ValueError(
                f"ParamSpec '{self.name}': default={self.default} is not finite."
            )
        if not (self.min <= self.default <= self.max):
            raise ValueError(
                f"ParamSpec '{self.name}': requires min <= default <= max, "
                f"but got min={self.min}, default={self.default}, max={self.max}."
            )

    def to_dict(self) -> dict:
        """Serialize to the exported JSON representation."""
        return {
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSpec":
        """Deserialize from the exported JSON representation."""
        return cls(
            name=d["name"],
            min=float(d["min"]),
            max=float(d["max"]),
            default=float(d["default"]),
        )
