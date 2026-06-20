#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import tempfile
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from typing import Sequence

from tensorboard.backend.event_processing import event_accumulator


_PARAMETRIC_BUCKETS = (
    "ESR_seen_audio_seen_param",
    "ESR_seen_audio_unseen_param",
    "ESR_unseen_audio_seen_param",
    "ESR_unseen_audio_unseen_param",
)


@dataclass(frozen=True)
class ScalarSummary:
    tag: str
    count: int
    best_index: int
    best_step: int
    best_value: float
    final_step: int
    final_value: float

    @property
    def min_to_final_ratio(self) -> float:
        if self.best_value == 0.0:
            return math.inf
        return self.final_value / self.best_value


@dataclass(frozen=True)
class RunSummary:
    label: str
    source: Path
    scalar_tags: tuple[str, ...]
    scalar_summaries: tuple[ScalarSummary, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize TensorBoard training logs from a raw event file, a "
            "directory containing one, or a zip archive."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more event files, event directories, or zip archives.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label for each path. Repeat once per input path.",
    )
    return parser.parse_args()


def _iter_event_files(path: Path) -> Iterable[Path]:
    if path.is_file() and path.name.startswith("events.out.tfevents."):
        yield path
        return
    if path.is_dir():
        for candidate in sorted(path.rglob("events.out.tfevents.*")):
            if candidate.is_file():
                yield candidate
        return
    if path.is_file() and path.suffix == ".zip":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    name = Path(member).name
                    if not name.startswith("events.out.tfevents."):
                        continue
                    extracted = tmpdir_path / name
                    extracted.write_bytes(archive.read(member))
                    yield extracted
        return
    raise FileNotFoundError(f"Unsupported log path: {path}")


def _load_event_accumulator(path: Path) -> event_accumulator.EventAccumulator:
    manager = tempfile.TemporaryDirectory() if path.is_file() and path.suffix == ".zip" else nullcontext()
    with manager as tmpdir:
        if tmpdir is None:
            event_path = next(_iter_event_files(path), None)
        else:
            event_path = None
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    name = Path(member).name
                    if not name.startswith("events.out.tfevents."):
                        continue
                    extracted = Path(tmpdir) / name
                    extracted.write_bytes(archive.read(member))
                    event_path = extracted
                    break
        if event_path is None:
            raise FileNotFoundError(f"No TensorBoard event file found in {path}")
        accumulator = event_accumulator.EventAccumulator(str(event_path))
        accumulator.Reload()
        return accumulator


def _summarize_tag(
    accumulator: event_accumulator.EventAccumulator, tag: str
) -> ScalarSummary:
    values = accumulator.Scalars(tag)
    best_index = min(range(len(values)), key=lambda i: values[i].value)
    best = values[best_index]
    final = values[-1]
    return ScalarSummary(
        tag=tag,
        count=len(values),
        best_index=best_index,
        best_step=best.step,
        best_value=best.value,
        final_step=final.step,
        final_value=final.value,
    )


def _preferred_tags(scalar_tags: Sequence[str]) -> list[str]:
    scalar_tag_set = set(scalar_tags)
    if all(tag in scalar_tag_set for tag in _PARAMETRIC_BUCKETS):
        return list(_PARAMETRIC_BUCKETS)

    preferred = [
        "ESR",
        "val_loss",
        "MSE",
        "MRSTFT",
        "ESR_packed_0",
        "ESR_packed_1",
    ]
    return [tag for tag in preferred if tag in scalar_tag_set]


def _summarize_run(path: Path, label: str | None = None) -> RunSummary:
    accumulator = _load_event_accumulator(path)
    scalar_tags = tuple(accumulator.Tags()["scalars"])
    tags = _preferred_tags(scalar_tags)
    summaries = tuple(_summarize_tag(accumulator, tag) for tag in tags)
    return RunSummary(
        label=label or path.stem,
        source=path,
        scalar_tags=scalar_tags,
        scalar_summaries=summaries,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = sum((x - x_mean) ** 2 for x in xs)
    y_denom = sum((y - y_mean) ** 2 for y in ys)
    if x_denom == 0.0 or y_denom == 0.0:
        return math.nan
    return numerator / math.sqrt(x_denom * y_denom)


def _print_run_summary(run: RunSummary, path: Path) -> None:
    print(f"# {run.label}")
    print(f"source: {path}")
    print(f"scalar_tags: {len(run.scalar_tags)}")
    for summary in run.scalar_summaries:
        print(
            f"{summary.tag}: best epoch_idx={summary.best_index} "
            f"step={summary.best_step} value={summary.best_value:.8g} "
            f"final={summary.final_value:.8g} "
            f"final/best={summary.min_to_final_ratio:.2f}"
        )
    scalar_lookup = {summary.tag: summary for summary in run.scalar_summaries}
    if all(tag in scalar_lookup for tag in _PARAMETRIC_BUCKETS):
        print("parametric_insight:")
        print(
            "  best deployment epoch_idx="
            f"{scalar_lookup['ESR_unseen_audio_unseen_param'].best_index}"
        )
        print(
            "  best primary epoch_idx="
            f"{scalar_lookup['ESR_unseen_audio_seen_param'].best_index}"
        )


def _print_parametric_relationships(path: Path) -> None:
    accumulator = _load_event_accumulator(path)
    scalar_tags = set(accumulator.Tags()["scalars"])
    if not all(tag in scalar_tags for tag in _PARAMETRIC_BUCKETS):
        return
    values = {
        tag: [entry.value for entry in accumulator.Scalars(tag)]
        for tag in _PARAMETRIC_BUCKETS
    }
    print("parametric_relationships:")
    print(
        "  corr(seen_audio_seen_param, unseen_audio_seen_param)="
        f"{_corr(values['ESR_seen_audio_seen_param'], values['ESR_unseen_audio_seen_param']):.6f}"
    )
    print(
        "  corr(seen_audio_unseen_param, unseen_audio_unseen_param)="
        f"{_corr(values['ESR_seen_audio_unseen_param'], values['ESR_unseen_audio_unseen_param']):.6f}"
    )


def main() -> int:
    args = _parse_args()
    if args.label and len(args.label) != len(args.paths):
        raise SystemExit("--label must be repeated exactly once per input path")

    labels = list(args.label) or [None] * len(args.paths)
    runs = [
        _summarize_run(Path(path).expanduser(), label=label)
        for path, label in zip(args.paths, labels)
    ]

    for index, (path, run) in enumerate(zip(args.paths, runs)):
        if index:
            print()
        resolved = Path(path).expanduser()
        _print_run_summary(run, resolved)
        _print_parametric_relationships(resolved)

    if len(runs) >= 2:
        print()
        print("# Comparison")
        for run in runs:
            best_esr = next(
                (
                    summary
                    for summary in run.scalar_summaries
                    if summary.tag in ("ESR_unseen_audio_unseen_param", "ESR")
                ),
                None,
            )
            if best_esr is None:
                continue
            print(
                f"{run.label}: best={best_esr.best_value:.8g} "
                f"epoch_idx={best_esr.best_index} final={best_esr.final_value:.8g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
