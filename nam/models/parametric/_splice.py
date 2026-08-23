"""
Supervised knob-move training from captures that share one input file.

Training windows carry a control setting that is constant, so nothing in training ever
shows a model what a control *change* looks like. The concat architectures then meet one
at inference with no supervision behind them: a dilated kernel spanning the change sees a
tap pattern it was never trained on, and a recurrent net's hidden state is out of an
equilibrium it was never taken out of during training.

Every capture in a run is reamped through the same input file, so at a given source
sample two captures are the device's answers to the *same* input at two settings. That
makes a supervised knob move available without capturing one: keep the input window
contiguous and assemble the *target* from different captures at the same source times,
with the control trajectory stepping at those same instants. The result is ground truth
for "the controls just changed and the device is already at the destination's steady
state" -- not the device's own analog settling, which is deliberately not being modeled.

The input is never spliced. A device's response to a joined signal is not the join of its
responses, so a joined input has no known target.

Only steps between captured settings are built. An intermediate setting has no capture
and therefore no target, so nothing here interpolates.

Indexing contract, with ``pad_start=False`` and a window opening at source sample
``abs0``:

- input frame ``j`` is source sample ``abs0 + j``
- output sample ``k`` is source sample ``abs0 + (receptive_field - 1) + k``
- ``length == (receptive_field - 1) + warmup + ny``, so the scored region opens at input
  frame ``(receptive_field - 1) + warmup``
- a step at scored sample ``s`` is a step at input frame ``s + (receptive_field - 1) + warmup``

An error in that last line trains the model to anticipate a knob move and still reports
plausible metrics, so it is asserted directly by the tests.
"""

import math as _math
from collections.abc import Sequence as _Sequence
from typing import NamedTuple as _NamedTuple
from typing import Optional as _Optional
from warnings import warn as _warn

import torch as _torch

from ._base import ParametricNet as _ParametricNet
from ._dataset import ParametricDataset as _ParametricDataset


# Samples of output compared when reporting inter-capture alignment. Ten seconds at 48k;
# a diagnostic does not need the whole capture, and this keeps the scan's cost flat.
_LAG_SCAN_SAMPLES = 480_000


class SpliceWindow(_NamedTuple):
    """The window one splice batch is run on; see :func:`splice_window`."""

    #: Samples the net is run on, history and warmup included.
    length: int
    #: Unscored samples between the receptive-field history and the scored region.
    warmup: int
    #: Scored output samples, the trailing slice of the window.
    ny: int
    #: Control steps within the window.
    steps: int
    #: Input-domain frame index of each step.
    step_offsets: tuple[int, ...]
    #: Scored-domain first sample of each post-step segment.
    segment_starts: tuple[int, ...]


class SpliceCases(_NamedTuple):
    """Which windows to build, resolved by :meth:`SpliceSource.build`."""

    #: (B,) source-file sample index at which each window opens.
    start: _torch.Tensor
    #: (B, steps + 1) capture index per segment; column 0 is the pre-step source.
    captures: _torch.Tensor


class _Capture(_NamedTuple):
    params: _torch.Tensor
    x: _torch.Tensor
    y: _torch.Tensor
    input_start: int
    label: str


def splice_window(
    net: _ParametricNet, *, steps: int, spacing: int, min_tail: int, settle: int
) -> SpliceWindow:
    """
    Size the window a net has to run to score ``steps`` control steps.

    The scored tail after the last step is ``max(receptive_field, min_tail)``. For a
    convolutional net a step more than one receptive field in the past no longer reaches
    the output, so scoring beyond that only dilutes the term with samples the ordinary
    loss already covers; at exactly the receptive field every scored sample straddles the
    step and the last one is a destination-only check. A recurrent net reports a
    receptive field of 1 and has no such bound, so ``min_tail`` supplies its settle
    length instead.

    ``settle`` is how much history the model gets at the pre-step setting. A convolutional
    net's ``receptive_field - 1`` of window history already is that history, so the
    request costs it nothing; a recurrent net has none and pays the difference as unscored
    warmup, which also covers the detached burn-in reported by ``training_warmup``.

    The first step lands on scored sample 0. Scored samples ahead of the first step are,
    for a causal convolution, identical to what the ordinary training loss already
    computes on the pre-step capture.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1; got {steps}")
    if steps > 1 and spacing < 1:
        raise ValueError(f"spacing must be at least 1; got {spacing}")
    if min_tail < 1:
        raise ValueError(f"min_tail must be at least 1; got {min_tail}")
    if settle < 0:
        raise ValueError(f"settle must not be negative; got {settle}")
    receptive_field = net.receptive_field
    tail = max(receptive_field, min_tail)
    ny = tail + (steps - 1) * spacing
    warmup = max(settle - (receptive_field - 1), net.training_warmup, 0)
    scored_start = receptive_field - 1 + warmup
    segment_starts = tuple(i * spacing for i in range(steps))
    return SpliceWindow(
        length=scored_start + ny,
        warmup=warmup,
        ny=ny,
        steps=steps,
        step_offsets=tuple(s + scored_start for s in segment_starts),
        segment_starts=segment_starts,
    )


def splice_output(
    net: _ParametricNet, x: _torch.Tensor, params: _torch.Tensor, window: SpliceWindow
) -> _torch.Tensor:
    """
    Run ``net`` over a built splice window and return the ``(B, ny)`` scored tail.

    Fed with ``pad_start=False``, so the model's history is inside the window rather than
    padded on afterwards -- which is what makes the target's source times line up with the
    output. The warmup samples are dropped here so the score is taken only where the
    gradient actually flows.
    """
    if not net.supports_param_trajectory:
        raise ValueError(
            f"{type(net).__name__} cannot evaluate a control that moves within one "
            "forward, so it cannot be trained on param-step splices"
        )
    y = net(x, params, pad_start=False)
    if y.shape[-1] < window.ny:
        raise RuntimeError(
            f"{type(net).__name__} returned {y.shape[-1]} samples for a splice window "
            f"sized to score {window.ny}; its history requirements are larger than "
            "receptive_field and training_warmup report"
        )
    return y[..., -window.ny :]


class SpliceSource:
    """
    Captures that share one input file, addressed by source-file sample index.

    All captures in a group must have been reamped through the same input at the same
    level and share one ``y_scale``: a per-capture output gain is indistinguishable from
    a real level control and would be spliced in as a step the device never made.
    ``start``/``stop``/``delay`` may differ freely -- they are resolved through each
    capture's ``input_start``.
    """

    def __init__(
        self,
        captures: _Sequence[_Capture],
        shared_range: tuple[int, int],
        mean_square: float,
        neighbor_order: _torch.Tensor,
    ):
        self._captures = tuple(captures)
        self._shared_range = shared_range
        self._mean_square = mean_square
        self._neighbor_order = neighbor_order
        self._params = _torch.stack([c.params for c in self._captures])

    @classmethod
    def from_datasets(
        cls,
        datasets: _Sequence[_ParametricDataset],
        net: _ParametricNet,
        *,
        dc_report_db: float = -60.0,
        lag_tolerance: int = 2,
    ) -> "SpliceSource":
        captures = tuple(
            _Capture(
                params=ds.params,
                x=ds.dataset.x,
                y=ds.dataset.y,
                input_start=ds.input_start,
                label=cls._label(i, ds, net),
            )
            for i, ds in enumerate(datasets)
        )
        if len(captures) < 2:
            raise ValueError(
                "Param-step splices need at least two captures to step between; got "
                f"{len(captures)}. Remove loss.param_step_splice or add captures."
            )
        for capture in captures:
            if capture.params.shape[-1] != net.param_dim:
                raise ValueError(
                    f"{capture.label} has {capture.params.shape[-1]} params but the net "
                    f"declares {net.param_dim}"
                )
        lo = max(c.input_start for c in captures)
        hi = min(c.input_start + len(c.x) for c in captures)
        if hi - lo < 1:
            raise ValueError(
                "Captures share no region of the input file; check start/stop/delay. "
                + ", ".join(
                    f"{c.label}: [{c.input_start}, {c.input_start + len(c.x)})"
                    for c in captures
                )
            )
        cls._validate_shared_input(captures, lo, hi)
        mean_square = float(
            _torch.stack(
                [cls._slice(c.y, c, lo, hi).square().mean() for c in captures]
            ).mean()
        )
        cls._report_dc(captures, lo, hi, mean_square, dc_report_db)
        cls._report_lag(captures, lo, hi, lag_tolerance)
        return cls(
            captures,
            (lo, hi),
            mean_square,
            cls._neighbor_order(captures, net),
        )

    @property
    def num_captures(self) -> int:
        return len(self._captures)

    @property
    def shared_range(self) -> tuple[int, int]:
        return self._shared_range

    @property
    def mean_square(self) -> float:
        """Mean of ``y**2`` over the shared region, pooled across captures.

        The energy scale the splice loss floors quiet windows against, expressed relative
        to the data so it survives output normalization and any ``y_scale``.
        """
        return self._mean_square

    def _start_span(self, window: SpliceWindow) -> int:
        """How far the window's opening sample can move, in samples."""
        low, high = self._shared_range
        span = high - low - window.length
        if span < 0:
            raise ValueError(
                f"Captures share only {high - low} samples of input, which is short of "
                f"the {window.length} a splice window needs. Widen the captures' "
                "start/stop range, or shrink min_tail and settle."
            )
        return span

    def sample_cases(
        self,
        n: int,
        window: SpliceWindow,
        *,
        neighbors: int,
        neighbor_probability: float,
        generator: _Optional[_torch.Generator] = None,
    ) -> SpliceCases:
        """
        Draw ``n`` random windows and control chains.

        Each step lands either on one of the current setting's ``neighbors`` nearest
        captures or on any other capture at all. Near steps are the elementary unit of a
        knob being turned and the common gesture, but their target difference is also the
        smallest relative to capture noise; a distant step is the worst case and gives the
        cleanest gradient, so both are drawn. Nearness is by rank rather than by distance
        so the mixture needs no tuning per capture set, and it collapses to the only
        available pair when there are two captures.
        """
        if n < 1:
            raise ValueError(f"n must be at least 1; got {n}")
        if neighbors < 1:
            raise ValueError(f"neighbors must be at least 1; got {neighbors}")
        if not 0.0 <= neighbor_probability <= 1.0:
            raise ValueError(
                f"neighbor_probability must be within [0, 1]; got {neighbor_probability}"
            )
        others = self.num_captures - 1
        near = min(neighbors, others)
        current = _torch.randint(self.num_captures, (n,), generator=generator)
        columns = [current]
        for _ in range(window.steps):
            order = self._neighbor_order[current]
            use_near = _torch.rand((n,), generator=generator) < neighbor_probability
            nxt = _torch.where(
                use_near,
                order.gather(1, _torch.randint(near, (n, 1), generator=generator))[:, 0],
                order.gather(1, _torch.randint(others, (n, 1), generator=generator))[:, 0],
            )
            columns.append(nxt)
            current = nxt
        low, _ = self._shared_range
        return SpliceCases(
            start=low
            + _torch.randint(self._start_span(window) + 1, (n,), generator=generator),
            captures=_torch.stack(columns, dim=1),
        )

    def enumerate_cases(self, n: int, window: SpliceWindow) -> SpliceCases:
        """
        The same ``n`` windows every time: evenly spaced starts against every ordered
        capture pair, cycled.

        A metric that has to be compared between epochs and between runs cannot be drawn
        fresh each time or the draw is most of what moves.
        """
        if n < 1:
            raise ValueError(f"n must be at least 1; got {n}")
        low, _ = self._shared_range
        span = self._start_span(window)
        pairs = [
            (a, b)
            for a in range(self.num_captures)
            for b in range(self.num_captures)
            if a != b
        ]
        start = _torch.tensor(
            [low + (span * i) // max(n - 1, 1) for i in range(n)], dtype=_torch.long
        )
        captures = _torch.tensor(
            [
                [pairs[i % len(pairs)][j % 2] for j in range(window.steps + 1)]
                for i in range(n)
            ],
            dtype=_torch.long,
        )
        return SpliceCases(start=start, captures=captures)

    def build(
        self, cases: SpliceCases, window: SpliceWindow, *, static: bool = False
    ) -> tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        """
        Materialize ``(x, params, target)`` as ``(B, length)``, ``(B, length, P)`` raw
        control trajectory, and ``(B, ny)``.

        ``static`` builds the control for the same windows: the setting is held at the
        final destination for the whole window and the target is that capture throughout.
        Scored against the spliced run it isolates the cost of the step from the model's
        static fit on the same audio. For a single step that is exact; for a chain it
        omits the earlier segments, which is why one step is the default.
        """
        batch_size = cases.start.shape[0]
        if cases.captures.shape != (batch_size, window.steps + 1):
            raise ValueError(
                f"Expected captures of shape {(batch_size, window.steps + 1)}; got "
                f"{tuple(cases.captures.shape)}"
            )
        scored_start = window.length - window.ny
        chain = (
            cases.captures[:, -1:].expand(-1, window.steps + 1)
            if static
            else cases.captures
        )

        frames = _torch.arange(window.length)
        offsets = _torch.tensor(window.step_offsets)
        segment_of_frame = (frames[None, :] >= offsets[:, None]).sum(dim=0)
        params = self._params[chain][:, segment_of_frame, :]

        x = _torch.empty((batch_size, window.length), dtype=self._captures[0].x.dtype)
        target = _torch.empty((batch_size, window.ny), dtype=self._captures[0].y.dtype)
        bounds = tuple(window.segment_starts) + (window.ny,)
        for b in range(batch_size):
            abs0 = int(cases.start[b])
            source = self._captures[int(chain[b, 0])]
            x[b] = self._slice(source.x, source, abs0, abs0 + window.length)
            for i in range(window.steps):
                capture = self._captures[int(chain[b, i + 1])]
                first = abs0 + scored_start + bounds[i]
                last = abs0 + scored_start + bounds[i + 1]
                target[b, bounds[i] : bounds[i + 1]] = self._slice(
                    capture.y, capture, first, last
                )
        return x, params, target

    @staticmethod
    def _label(index: int, dataset: _ParametricDataset, net: _ParametricNet) -> str:
        """Name a capture by its control setting, which is how these get reasoned about.

        The audio path would be the obvious name, but ``Dataset.parse_config`` consumes
        ``y_path`` without passing it on, so a capture built from a config carries none.
        """
        setting = ", ".join(
            f"{spec.name}={float(value):g}"
            for spec, value in zip(net.param_specs, dataset.params)
        )
        path = getattr(dataset.dataset, "_y_path", None)
        return f"capture {index} ({setting})" + (f" {path}" if path else "")

    @staticmethod
    def _slice(
        audio: _torch.Tensor, capture: _Capture, first: int, last: int
    ) -> _torch.Tensor:
        return audio[first - capture.input_start : last - capture.input_start]

    @classmethod
    def _validate_shared_input(
        cls, captures: _Sequence[_Capture], lo: int, hi: int
    ) -> None:
        """
        Every capture must be the device's answer to the same input samples.

        One comparison covers a differing ``x_path``, a differing ``input_gain`` (which
        means the captures were driven at different levels, so their outputs are answers
        to different signals), a resample mismatch, and a wrong ``input_start``. It is a
        hard failure rather than a warning because a splice built on it has no valid
        target at all.
        """
        reference = cls._slice(captures[0].x, captures[0], lo, hi)
        reference_rms = float(reference.square().mean().sqrt())
        for capture in captures[1:]:
            other = cls._slice(capture.x, capture, lo, hi)
            if _torch.equal(reference, other):
                continue
            other_rms = float(other.square().mean().sqrt())
            deviation = float((reference - other).abs().max())
            ratio = other_rms / reference_rms if reference_rms > 0.0 else float("nan")
            raise ValueError(
                f"{capture.label} does not share an input with {captures[0].label} over "
                f"the common region [{lo}, {hi}): max deviation {deviation:.3e}, RMS "
                f"ratio {ratio:.4f}. A clean ratio means the two captures declare "
                "different input_gain; otherwise they are different input files or "
                "misaligned. Param-step splices require one shared input."
            )

    @classmethod
    def _report_dc(
        cls,
        captures: _Sequence[_Capture],
        lo: int,
        hi: int,
        mean_square: float,
        dc_report_db: float,
    ) -> None:
        """
        A DC difference between captures becomes an instantaneous DC step in every
        spliced target, which no device produces -- it slews over its coupling time
        constant. It is reported rather than subtracted: removing it would make the splice
        target disagree with the target the ordinary loss is fitting on the same captures.
        """
        offsets = [float(cls._slice(c.y, c, lo, hi).mean()) for c in captures]
        worst = max(offsets) - min(offsets)
        scale = _math.sqrt(mean_square)
        if scale <= 0.0 or worst <= 0.0:
            return
        worst_db = 20.0 * _math.log10(worst / scale)
        if worst_db < dc_report_db:
            return
        _warn(
            f"Captures differ in DC offset by {worst_db:.1f} dB relative to their own "
            "level. Every param-step splice will teach an instantaneous DC step that the "
            "device does not make. Per capture: "
            + ", ".join(f"{c.label}={o:+.3e}" for c, o in zip(captures, offsets))
        )

    @classmethod
    def _report_lag(
        cls, captures: _Sequence[_Capture], lo: int, hi: int, lag_tolerance: int
    ) -> None:
        """
        A residual alignment error between captures is a broadband step at every splice
        boundary -- exactly the artifact this is meant to remove.

        A nonzero peak is not on its own a fault: a tone control genuinely changes group
        delay, so settings really do sit at different delays. The numbers are printed for
        that reason; a uniform offset across every capture is a reamp clocking problem
        while a setting-dependent spread is the device. Only a peak past ``lag_tolerance``
        is treated as broken alignment.
        """
        search = lag_tolerance + 2
        span = min(hi - lo, _LAG_SCAN_SAMPLES)
        if span <= 4 * search:
            return
        mid = (lo + hi) // 2
        first, last = mid - span // 2, mid - span // 2 + span
        reference = cls._slice(captures[0].y, captures[0], first + search, last - search)
        reference = reference - reference.mean()
        norm = float(reference.square().sum().sqrt())
        if norm <= 0.0:
            return
        lags = []
        for capture in captures[1:]:
            scores = []
            for lag in range(-search, search + 1):
                other = cls._slice(
                    capture.y, capture, first + search + lag, last - search + lag
                )
                other = other - other.mean()
                other_norm = float(other.square().sum().sqrt())
                scores.append(
                    float(reference.dot(other)) / (norm * other_norm)
                    if other_norm > 0.0
                    else 0.0
                )
            lags.append((capture.label, cls._interpolated_peak(scores, search)))
        worst = max(abs(lag) for _, lag in lags)
        if worst >= lag_tolerance:
            raise ValueError(
                f"Captures are misaligned by up to {worst:.2f} samples, past the "
                f"tolerance of {lag_tolerance}: "
                + ", ".join(f"{label}={lag:+.2f}" for label, lag in lags)
            )
        if worst >= 1.0:
            _warn(
                "Captures show an output delay spread of up to "
                f"{worst:.2f} samples relative to {captures[0].label}: "
                + ", ".join(f"{label}={lag:+.2f}" for label, lag in lags)
                + ". A spread that tracks the controls is the device; a uniform offset "
                "is a reamp alignment problem and will be spliced in as a click."
            )

    @staticmethod
    def _interpolated_peak(scores: _Sequence[float], search: int) -> float:
        best = max(range(len(scores)), key=lambda i: scores[i])
        lag = float(best - search)
        if 0 < best < len(scores) - 1:
            left, peak, right = scores[best - 1], scores[best], scores[best + 1]
            denominator = left - 2.0 * peak + right
            if denominator != 0.0:
                lag += 0.5 * (left - right) / denominator
        return lag

    @staticmethod
    def _neighbor_order(
        captures: _Sequence[_Capture], net: _ParametricNet
    ) -> _torch.Tensor:
        """
        ``(N, N - 1)`` of every other capture, nearest first.

        Distance is taken in the net's own encoded control space, which is the space the
        conditioning actually reaches the model in: continuous controls are already mapped
        onto a common normalized range there, so they compare without a separate
        normalization step, and a switch flip is a fixed distance however its indices are
        numbered.
        """
        with _torch.no_grad():
            encoded = net._encode_params(
                _torch.stack([c.params for c in captures])
            ).cpu()
        distance = _torch.cdist(encoded, encoded)
        distance.fill_diagonal_(float("inf"))
        return distance.argsort(dim=1)[:, : len(captures) - 1].contiguous()
