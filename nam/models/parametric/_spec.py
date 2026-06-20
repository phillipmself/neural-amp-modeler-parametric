"""
ParamSpec: a self-describing parameter specification for parametric NAM models.

Each ParamSpec captures everything the model and downstream tooling need to know
about one continuous control parameter: its name (for UI display), its numeric
range (for plugin/UI clamping and normalization), and its default value (for
export snapshots and loudness normalization).

Design note: min/max are metadata for downstream consumers (plugins, UIs) and
define how raw knob values are normalized before reaching the adapter. The net
consumes default values positionally at export time, exactly as the old
nominal_params did.
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
    - min < max.
    - min <= default <= max.
    """

    NORMALIZATION = "min_max_signed"
    NORMALIZED_MIN = -1.0
    NORMALIZED_MAX = 1.0

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
        if not self.min < self.max:
            raise ValueError(
                f"ParamSpec '{self.name}': requires min < max so the parameter "
                f"has a real span, but got min={self.min}, max={self.max}."
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
            "input_normalization": self.NORMALIZATION,
            "normalized_min": self.NORMALIZED_MIN,
            "normalized_max": self.NORMALIZED_MAX,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSpec":
        """Deserialize from the exported JSON representation."""
        normalization = d.get("input_normalization")
        if normalization is not None and normalization != cls.NORMALIZATION:
            raise ValueError(
                "Unsupported param input normalization "
                f"{normalization!r}; expected {cls.NORMALIZATION!r}."
            )
        normalized_min = d.get("normalized_min")
        if normalized_min is not None and float(normalized_min) != cls.NORMALIZED_MIN:
            raise ValueError(
                "Unsupported normalized_min "
                f"{normalized_min!r}; expected {cls.NORMALIZED_MIN}."
            )
        normalized_max = d.get("normalized_max")
        if normalized_max is not None and float(normalized_max) != cls.NORMALIZED_MAX:
            raise ValueError(
                "Unsupported normalized_max "
                f"{normalized_max!r}; expected {cls.NORMALIZED_MAX}."
            )
        return cls(
            name=d["name"],
            min=float(d["min"]),
            max=float(d["max"]),
            default=float(d["default"]),
        )

    @property
    def center(self) -> float:
        return 0.5 * (self.min + self.max)

    @property
    def half_range(self) -> float:
        return 0.5 * (self.max - self.min)
