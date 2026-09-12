"""Produce per-fold error evidence for the selected M1 agriculture forecasts.

The aggregate MAPE/RMSE figures in ``docs/EVALUATION.md`` can conceal one bad
forecast.  This module exposes the three held-out, rolling-origin predictions
behind the selected annual-naive models without changing model artifacts.

    python -m eval.agriculture_forecast_errors

The default JSON output is deliberately local and gitignored.  The report is
derived from dated source snapshots, not the mutable development database.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ceynex.models.agriculture.baseline import AnnualNaiveModel
from ceynex.models.agriculture.evaluation import load_agriculture_series
from eval.backtest import Fold, rolling_origin_folds


@dataclass(frozen=True)
class ForecastError:
    """One held-out observation and its honest one-step forecast."""

    item: str
    target: str
    unit: str
    fold: int
    train_end_period: int
    test_period: int
    actual: float
    prediction: float
    absolute_error: float
    absolute_percentage_error: float | None
    lower_80: float
    upper_80: float
    covered_by_80_interval: bool


def _model(item: str, target: str, unit: str) -> AnnualNaiveModel:
    return AnnualNaiveModel(sector="agriculture", item=item, target=target, unit=unit)


def _error_rows(item: str, target: str, unit: str, folds: list[Fold]) -> list[ForecastError]:
    rows: list[ForecastError] = []
    for fold in folds:
        # M1's evaluation is explicitly one-year-ahead.  Keep the assertion so
        # a future horizon change cannot silently turn this into partial output.
        if len(fold.actual) != 1:
            raise ValueError("agriculture forecast-error report expects one-year-ahead folds")
        actual = fold.actual[0]
        prediction = fold.predicted[0]
        rows.append(
            ForecastError(
                item=item,
                target=target,
                unit=unit,
                fold=fold.index + 1,
                train_end_period=fold.train_periods[-1],
                test_period=fold.test_periods[0],
                actual=actual,
                prediction=prediction,
                absolute_error=abs(actual - prediction),
                absolute_percentage_error=None if actual == 0 else abs((actual - prediction) / actual),
                lower_80=fold.lower[0],
                upper_80=fold.upper[0],
                covered_by_80_interval=fold.lower[0] <= actual <= fold.upper[0],
            )
        )
    return rows


def analyse_forecast_errors(raw_root: Path | None = None) -> list[ForecastError]:
    """Return fold-level errors for the two selected annual-naive M1 models."""
    series = load_agriculture_series(raw_root)
    specifications = (
        ("tea", "export_volume", "kg", "tea_volume"),
        ("cinnamon", "producer_price", "USD/kg", "cinnamon_price"),
    )
    results: list[ForecastError] = []
    for item, target, unit, key in specifications:
        model = _model(item, target, unit)
        folds = rolling_origin_folds(model, series[key])
        results.extend(_error_rows(item, target, unit, folds))
    return results


def summary(rows: list[ForecastError]) -> list[dict[str, Any]]:
    """Summarise fold errors without reimplementing or relabelling MAPE/RMSE."""
    grouped: dict[tuple[str, str, str], list[ForecastError]] = {}
    for row in rows:
        grouped.setdefault((row.item, row.target, row.unit), []).append(row)
    return [
        {
            "item": item,
            "target": target,
            "unit": unit,
            "held_out_observations": len(group),
            "largest_absolute_error": max(entry.absolute_error for entry in group),
            "largest_absolute_percentage_error": max(
                entry.absolute_percentage_error or 0.0 for entry in group
            ),
            "interval_coverage": sum(entry.covered_by_80_interval for entry in group) / len(group),
        }
        for (item, target, unit), group in sorted(grouped.items())
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-fold errors for selected agriculture forecasts.")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/results/agriculture_forecast_errors.json"),
        help="Local JSON record path; eval/results is gitignored.",
    )
    args = parser.parse_args(argv)
    rows = analyse_forecast_errors(args.raw_root)
    payload = {"fold_errors": [asdict(row) for row in rows], "summary": summary(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
