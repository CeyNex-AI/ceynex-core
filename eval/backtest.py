"""Implements SRS 3.1.10 — the one rolling-origin backtest harness.

    make backtest SECTOR=agriculture ITEM=cinnamon

Three members train models. If each computes their own MAPE, the three numbers
in the evaluation report cannot be compared, and the report's whole claim is a
comparison. So there is exactly one implementation and every model's
`backtest()` delegates to it.

**Expanding window, never a random split.** Fold *k* trains on everything up to
period *t* and predicts *t+1…t+h*; the next fold moves the origin forward and
trains on more. A random train/test split on a time series lets the model see
the future, and produces a beautiful MAPE that means nothing. `_folds` asserts
the ordering rather than trusting it.

Three metrics, because each hides a different failure:

- **MAPE** — average error size, scale-free, so cinnamon and apparel compare.
  Blind to whether the uncertainty was honest.
- **RMSE** — in the series' own units, and punishes large misses harder than
  MAPE, so one catastrophic year cannot be averaged away.
- **coverage** — the fraction of actuals that landed inside the 80% interval.
  This is the one that catches a model with a great MAPE and dishonest bands.
  Read it against 0.80: much lower means overconfident, much higher means the
  intervals are so wide they say nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_FOLDS = 3
DEFAULT_HORIZON = 1
# Below this a fold's training slice cannot support a model, so folds are
# dropped rather than fitted on three points and reported as if meaningful.
MIN_TRAIN = 5


class BacktestError(RuntimeError):
    """Raised when a series cannot support an honest backtest."""


@dataclass(frozen=True)
class Fold:
    """One origin: train on `train_periods`, predict `test_periods`."""

    index: int
    train_periods: list[int]
    test_periods: list[int]
    predicted: list[float]
    actual: list[float]
    lower: list[float]
    upper: list[float]


def rolling_origin(
    model: Any,
    frame: pd.DataFrame,
    *,
    folds: int = DEFAULT_FOLDS,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, float]:
    """Backtest a model on its own history. Returns mape, rmse, coverage.

    `model` is refitted from scratch on each fold — a clone built from
    `describe_params()`, never the passed instance, which would carry the full
    series it was already fitted on straight into the training slice.
    """
    results = rolling_origin_folds(model, frame, folds=folds, horizon=horizon)
    return summarize(results)


def rolling_origin_folds(
    model: Any,
    frame: pd.DataFrame,
    *,
    folds: int = DEFAULT_FOLDS,
    horizon: int = DEFAULT_HORIZON,
) -> list[Fold]:
    """The per-fold detail behind `rolling_origin`."""
    periods, values = _series(model, frame)
    splits = _folds(len(values), folds=folds, horizon=horizon)
    if not splits:
        raise BacktestError(
            f"{len(values)} observations cannot support {folds} folds at horizon "
            f"{horizon} with a minimum training window of {MIN_TRAIN}"
        )

    completed: list[Fold] = []
    for index, (train_end, test_end) in enumerate(splits):
        clone = _clone(model)
        train = pd.DataFrame(
            {"period": periods[:train_end], "value": values[:train_end]}
        )
        try:
            clone.fit(train)
            points = clone.predict(test_end - train_end)
        except Exception as exc:  # noqa: BLE001 - a fold that will not fit is data, not a crash
            log.warning("fold %d did not fit (%s); excluded from the metrics", index, exc)
            continue

        completed.append(
            Fold(
                index=index,
                train_periods=periods[:train_end],
                test_periods=periods[train_end:test_end],
                predicted=[float(p["point"]) for p in points],
                actual=[float(v) for v in values[train_end:test_end]],
                lower=[float(p["lower"]) for p in points],
                upper=[float(p["upper"]) for p in points],
            )
        )

    if not completed:
        raise BacktestError("no fold could be fitted; the series is too short or too irregular")
    return completed


def summarize(folds: list[Fold]) -> dict[str, float]:
    """Pool every fold's predictions into one set of metrics."""
    predicted = [p for fold in folds for p in fold.predicted]
    actual = [a for fold in folds for a in fold.actual]
    lower = [x for fold in folds for x in fold.lower]
    upper = [x for fold in folds for x in fold.upper]

    return {
        "mape": mape(actual, predicted),
        "rmse": rmse(actual, predicted),
        "coverage": coverage(actual, lower, upper),
        "folds": float(len(folds)),
        "observations": float(len(actual)),
    }


# --- metrics -------------------------------------------------------------


def mape(actual: list[float], predicted: list[float]) -> float:
    """Mean absolute percentage error, as a fraction.

    Periods where the actual is zero are skipped: the percentage error is
    undefined there, and the usual dodge of adding epsilon to the denominator
    invents an enormous error term out of a rounding constant.
    """
    pairs = [(a, p) for a, p in zip(actual, predicted, strict=True) if a != 0]
    if not pairs:
        return float("nan")
    return sum(abs((a - p) / a) for a, p in pairs) / len(pairs)


def rmse(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return float("nan")
    squares = [(a - p) ** 2 for a, p in zip(actual, predicted, strict=True)]
    return math.sqrt(sum(squares) / len(squares))


def coverage(actual: list[float], lower: list[float], upper: list[float]) -> float:
    """Fraction of actuals inside the predicted interval. Compare against 0.80."""
    if not actual:
        return float("nan")
    inside = sum(1 for a, lo, hi in zip(actual, lower, upper, strict=True) if lo <= a <= hi)
    return inside / len(actual)


# --- splitting -----------------------------------------------------------


def _folds(n: int, *, folds: int, horizon: int) -> list[tuple[int, int]]:
    """Expanding-window origins as `(train_end, test_end)` index pairs.

    Every pair satisfies `train_end <= test_start`, which is the property that
    makes this a time-series backtest rather than a leak.
    """
    splits: list[tuple[int, int]] = []
    for k in range(folds, 0, -1):
        train_end = n - k * horizon
        test_end = train_end + horizon
        if train_end < MIN_TRAIN or test_end > n:
            continue
        assert train_end <= test_end, "a fold trained past its own test window"
        splits.append((train_end, test_end))
    return splits


def _series(model: Any, frame: pd.DataFrame) -> tuple[list[int], list[float]]:
    extract = getattr(model, "_series", None)
    if callable(extract):
        return extract(frame)
    return (
        [int(p) for p in frame["period"].tolist()],
        [float(v) for v in frame["value"].tolist()],
    )


def _clone(model: Any) -> Any:
    """A fresh, unfitted model with the same hyperparameters."""
    describe = getattr(model, "describe_params", None)
    params = dict(describe()) if callable(describe) else {}
    params.setdefault("sector", getattr(model, "sector", "unknown"))
    params.setdefault("item", getattr(model, "item", "unknown"))
    params.setdefault("target", getattr(model, "target", "export_value_usd"))
    return type(model)(**params)


# --- CLI -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rolling-origin backtest for a registered item.")
    parser.add_argument("--sector", required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--target", default="export_value_usd")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--model",
        default="timeseries",
        choices=["timeseries", "gbm"],
        help="which family to fit when nothing is registered yet",
    )
    parser.add_argument("--register", action="store_true", help="save the fitted model with its metrics")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from ceynex.data.reader import DatasetUnavailableError, annual_series

    try:
        frame = annual_series(args.item, sector=args.sector, target=args.target)
    except DatasetUnavailableError as exc:
        print(f"could not read the dataset: {exc}", file=sys.stderr)
        return 2

    if frame.empty:
        print(f"no annual {args.target} rows for {args.sector}/{args.item}", file=sys.stderr)
        return 2

    model = _build(args.model, sector=args.sector, item=args.item, target=args.target)
    try:
        model.fit(frame)
        metrics = rolling_origin(model, frame, folds=args.folds, horizon=args.horizon)
    except (BacktestError, ValueError) as exc:
        print(f"backtest failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"sector": args.sector, "item": args.item, **metrics}, indent=2))
    _interpret(metrics)

    if args.register:
        from ceynex.models import registry

        metadata = registry.save(model, training_rows=len(frame), metrics=metrics)
        print(f"registered {metadata.model_id}")
    return 0


def _build(family: str, **kwargs: Any) -> Any:
    if family == "gbm":
        from ceynex.models.gbm import GradientBoostedModel

        return GradientBoostedModel(**kwargs)
    from ceynex.models.timeseries import TimeSeriesModel

    return TimeSeriesModel(**kwargs)


def _interpret(metrics: dict[str, float]) -> None:
    """Say what the coverage number means, since it is the one people misread."""
    achieved = metrics.get("coverage")
    if achieved is None or math.isnan(achieved):
        return
    if achieved < 0.5:
        print(
            f"\ncoverage {achieved:.2f} against a nominal 0.80: the intervals are "
            "too narrow, so this model is more confident than its errors justify.",
            file=sys.stderr,
        )
    elif achieved > 0.95:
        print(
            f"\ncoverage {achieved:.2f} against a nominal 0.80: the intervals are so "
            "wide they are unlikely to be useful for a decision.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
