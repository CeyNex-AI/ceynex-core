"""Implements SRS 3.1.10 and 3.3.4 — agriculture source checks and backtests.

The source series are read directly from M1's dated raw snapshots, keeping the
model experiment reproducible without depending on a mutable development
database.  Both are annual and below the project's 40-observation threshold,
so the evaluation treats simple baselines as the default rather than treating
complexity as an achievement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ceynex.data.connectors._snapshots import latest_snapshot_dir, resolve_snapshot_file
from ceynex.models.agriculture.baseline import AnnualNaiveModel
from ceynex.models.gbm import GradientBoostedModel
from ceynex.models.registry import DEFAULT_INTERVAL_LEVEL, ModelMetadata, save
from ceynex.models.timeseries import TimeSeriesModel
from ceynex.settings import REPO_ROOT, data_dir
from eval.backtest import DEFAULT_FOLDS, DEFAULT_HORIZON, rolling_origin

MIN_COMPLEX_OBSERVATIONS = 40
GBM_RELATIVE_IMPROVEMENT = 0.05

MODEL_SOURCES = {
    "tea": "Sri Lanka Tea Board annual total exports",
    "cinnamon": "FAOSTAT annual Sri Lanka cinnamon producer price (USD/kg)",
}


@dataclass(frozen=True)
class SeriesSufficiency:
    """Auditable observation-count decision for one model target."""

    item: str
    target: str
    unit: str
    observations: int
    period_start: int
    period_end: int
    frequency: str
    below_complex_model_threshold: bool
    source: str


@dataclass(frozen=True)
class ModelEvaluation:
    """Metrics from one expanding-window, one-year-ahead backtest."""

    item: str
    model: str
    metrics: dict[str, float]
    selected: bool


def agriculture_raw_dir(raw_root: Path | None = None) -> Path:
    """Locate dated M1 raw data, allowing an explicit test/deployment override."""
    if raw_root is not None:
        return Path(raw_root)
    override = os.environ.get("CEYNEX_AGRICULTURE_RAW_DIR")
    candidates = [
        Path(override) if override else None,
        REPO_ROOT.parent / "data" / "raw",
        data_dir() / "raw",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(f"agriculture raw data not found; searched {searched}")


def tea_export_volume_series(workbook_path: Path) -> pd.DataFrame:
    """Return annual Tea Board total export volume in canonical kg."""
    workbook = resolve_snapshot_file(Path(workbook_path), "tea_annual_production_exports_2011_2025.xlsx")
    frame = pd.read_excel(workbook, sheet_name="Exports", header=3)
    required = {"year", "total_exports_mt"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Tea Board export sheet missing columns: {sorted(missing)}")
    series = frame.loc[:, ["year", "total_exports_mt"]].copy()
    series["period"] = pd.to_numeric(series["year"], errors="raise").astype("int64")
    series["value"] = pd.to_numeric(series["total_exports_mt"], errors="raise") * 1000.0
    return series.loc[:, ["period", "value"]].dropna().sort_values("period", ignore_index=True)


def cinnamon_price_series(producer_prices_path: Path) -> pd.DataFrame:
    """Return annual FAOSTAT cinnamon producer prices in canonical USD/kg."""
    frame = pd.read_csv(producer_prices_path, encoding="latin1")
    required = {"Area", "Area Code (M49)", "Item", "Element", "Year", "Value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"FAOSTAT producer-price CSV missing columns: {sorted(missing)}")
    series = frame.loc[
        frame["Area"].astype("string").str.strip().eq("Sri Lanka")
        & pd.to_numeric(frame["Area Code (M49)"], errors="coerce").eq(144)
        & frame["Item"].eq("Cinnamon and cinnamon-tree flowers, raw")
        & frame["Element"].eq("Producer Price (USD/tonne)"),
        ["Year", "Value"],
    ].copy()
    series["period"] = pd.to_numeric(series["Year"], errors="raise").astype("int64")
    series["value"] = pd.to_numeric(series["Value"], errors="raise") / 1000.0
    return series.loc[:, ["period", "value"]].dropna().sort_values("period", ignore_index=True)


def load_agriculture_series(raw_root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the two annual model targets from their dated raw-source folders."""
    root = agriculture_raw_dir(raw_root)
    tea_workbook = root / "tea_board"
    price_snapshot = latest_snapshot_dir(root / "faostat")
    producer_prices = price_snapshot / "FAOSTAT producer prices.csv"
    if not producer_prices.is_file():
        raise FileNotFoundError(producer_prices)
    return {
        "tea_volume": tea_export_volume_series(tea_workbook),
        "cinnamon_price": cinnamon_price_series(producer_prices),
    }


def sufficiency_report(raw_root: Path | None = None) -> list[SeriesSufficiency]:
    """Count the real series before choosing the annual short-series strategy."""
    series = load_agriculture_series(raw_root)
    details = (
        ("tea", "export_volume", "kg", "tea_volume", "Tea Board annual total exports"),
        ("cinnamon", "producer_price", "USD/kg", "cinnamon_price", "FAOSTAT annual USD producer price"),
    )
    return [
        SeriesSufficiency(
            item=item,
            target=target,
            unit=unit,
            observations=len(series[key]),
            period_start=int(series[key]["period"].min()),
            period_end=int(series[key]["period"].max()),
            frequency="A",
            below_complex_model_threshold=len(series[key]) < MIN_COMPLEX_OBSERVATIONS,
            source=source,
        )
        for item, target, unit, key, source in details
    ]


def _evaluate(model: Any, frame: pd.DataFrame, *, item: str, name: str) -> ModelEvaluation:
    model.fit(frame)
    metrics = rolling_origin(model, frame, folds=DEFAULT_FOLDS, horizon=DEFAULT_HORIZON)
    point = model.predict(1)[0]
    assert point["lower"] <= point["point"] <= point["upper"], "forecast interval must contain point"
    return ModelEvaluation(item=item, model=name, metrics=metrics, selected=False)


def _select_best(evaluations: list[ModelEvaluation]) -> list[ModelEvaluation]:
    best = min(evaluations, key=lambda result: result.metrics["mape"])
    return [
        ModelEvaluation(result.item, result.model, result.metrics, result.model == best.model)
        for result in evaluations
    ]


def _should_select_gbm(gbm: ModelEvaluation, simple: list[ModelEvaluation]) -> bool:
    """Require a material (5%) MAPE gain before choosing GBM on a short series."""
    best_simple = min(result.metrics["mape"] for result in simple)
    return gbm.metrics["mape"] <= best_simple * (1.0 - GBM_RELATIVE_IMPROVEMENT)


def evaluate_agriculture_models(raw_root: Path | None = None) -> list[ModelEvaluation]:
    """Evaluate annual baselines first, selecting GBM only for a real gain."""
    series = load_agriculture_series(raw_root)

    tea = series["tea_volume"]
    tea_candidates = [
        _evaluate(AnnualNaiveModel(sector="agriculture", item="tea", target="value", unit="kg"), tea, item="tea", name="annual_naive"),
        _evaluate(AnnualNaiveModel(sector="agriculture", item="tea", target="value", unit="kg", strategy="drift"), tea, item="tea", name="annual_drift"),
        _evaluate(TimeSeriesModel(sector="agriculture", item="tea", target="value", unit="kg"), tea, item="tea", name="sarima_or_ets"),
    ]

    cinnamon = series["cinnamon_price"]
    cinnamon_simple = [
        _evaluate(AnnualNaiveModel(sector="agriculture", item="cinnamon", target="value", unit="USD/kg"), cinnamon, item="cinnamon", name="annual_naive"),
        _evaluate(AnnualNaiveModel(sector="agriculture", item="cinnamon", target="value", unit="USD/kg", strategy="drift"), cinnamon, item="cinnamon", name="annual_drift"),
        _evaluate(TimeSeriesModel(sector="agriculture", item="cinnamon", target="value", unit="USD/kg"), cinnamon, item="cinnamon", name="sarima_or_ets"),
    ]
    gbm = _evaluate(
        GradientBoostedModel(
            sector="agriculture",
            item="cinnamon",
            target="value",
            unit="USD/kg",
            params={"n_jobs": 1},
        ),
        cinnamon,
        item="cinnamon",
        name="gbm",
    )
    if _should_select_gbm(gbm, cinnamon_simple):
        cinnamon_candidates = [*cinnamon_simple, gbm]
        cinnamon_results = [
            ModelEvaluation(result.item, result.model, result.metrics, result.model == "gbm")
            for result in cinnamon_candidates
        ]
    else:
        cinnamon_results = [*_select_best(cinnamon_simple), gbm]

    return [*_select_best(tea_candidates), *cinnamon_results]


def current_git_sha() -> str:
    """Return the exact source revision used for a registry artifact."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot determine Git SHA; register from a Git checkout") from exc
    git_sha = result.stdout.strip()
    if not _is_git_sha(git_sha):
        raise RuntimeError(f"git rev-parse returned an invalid SHA: {git_sha!r}")
    return git_sha


def _is_git_sha(value: str) -> bool:
    return len(value) == 40 and all(char in "0123456789abcdef" for char in value.lower())


def require_clean_git_worktree() -> None:
    """Prevent artifacts claiming a SHA that does not contain their code."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify Git worktree; commit before model registration") from exc
    if result.stdout.strip():
        raise RuntimeError(
            "working tree has uncommitted changes; commit them before registering a model so its Git SHA is reproducible"
        )


def _required_registration_metadata(metadata: ModelMetadata) -> None:
    """Fail closed if a final M1 artifact lacks a field promised in its handoff."""
    required = {
        "training_rows": metadata.training_rows,
        "training_window": metadata.training_window,
        "source": metadata.source,
        "metrics": metadata.metrics,
        "interval_level": metadata.interval_level,
        "git_sha": metadata.git_sha,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"incomplete agriculture registry metadata: {', '.join(missing)}")
    metric_names = {"mape", "rmse", "coverage", "folds"}
    absent_metrics = metric_names.difference(metadata.metrics or {})
    if absent_metrics:
        raise RuntimeError(f"registry metrics missing: {sorted(absent_metrics)}")
    if metadata.interval_level != DEFAULT_INTERVAL_LEVEL:
        raise RuntimeError("M1 agriculture registry artifacts must have 80% intervals")


def register_selected_models(
    raw_root: Path | None = None,
    *,
    git_sha: str | None = None,
    evaluations: list[ModelEvaluation] | None = None,
) -> list[ModelMetadata]:
    """Fit and register the two selected honest annual-naïve agriculture models.

    The caller is expected to call :func:`require_clean_git_worktree` when this
    creates production artifacts.  `git_sha` is injectable solely for isolated
    tests and controlled build systems.
    """
    series = load_agriculture_series(raw_root)
    evaluations = evaluations if evaluations is not None else evaluate_agriculture_models(raw_root)
    selected = {(result.item, result.model): result for result in evaluations if result.selected}
    expected = {("tea", "annual_naive"), ("cinnamon", "annual_naive")}
    if set(selected) != expected:
        raise RuntimeError(
            "registration requires annual_naive to be selected for tea and cinnamon; "
            f"received {sorted(selected)}"
        )
    resolved_sha = git_sha or current_git_sha()
    if not _is_git_sha(resolved_sha):
        raise ValueError("git_sha must be a 40-character hexadecimal commit SHA")
    specifications = (
        ("tea", "export_volume", "kg", "tea_volume", None),
        (
            "cinnamon",
            "producer_price",
            "USD/kg",
            "cinnamon_price",
            "Liyanage/Silva purchasing-price data unavailable; FAOSTAT annual fallback used.",
        ),
    )

    registered: list[ModelMetadata] = []
    for item, target, unit, series_key, limitation in specifications:
        frame = series[series_key]
        model = AnnualNaiveModel(sector="agriculture", item=item, target=target, unit=unit).fit(frame)
        forecast = model.predict(1)[0]
        if not forecast["lower"] <= forecast["point"] <= forecast["upper"]:
            raise RuntimeError(f"{item} forecast interval does not contain its point forecast")
        metadata = save(
            model,
            training_rows=len(frame),
            training_window={"period_start": int(frame["period"].min()), "period_end": int(frame["period"].max())},
            source=MODEL_SOURCES[item],
            metrics=dict(selected[(item, "annual_naive")].metrics),
            interval_level=DEFAULT_INTERVAL_LEVEL,
            git_sha=resolved_sha,
            notes=(
                "Selected after three-fold expanding-window rolling-origin backtesting, one-year horizon."
                + (f" {limitation}" if limitation else "")
            ),
        )
        _required_registration_metadata(metadata)
        registered.append(metadata)
    return registered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate annual M1 agriculture source series.")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument(
        "--register",
        action="store_true",
        help="register the selected models; requires a clean committed Git worktree",
    )
    args = parser.parse_args(argv)
    report = sufficiency_report(args.raw_root)
    evaluations = evaluate_agriculture_models(args.raw_root)
    registrations: list[ModelMetadata] = []
    if args.register:
        require_clean_git_worktree()
        registrations = register_selected_models(
            args.raw_root,
            git_sha=current_git_sha(),
            evaluations=evaluations,
        )
    print(
        json.dumps(
            {
                "sufficiency": [asdict(row) for row in report],
                "evaluations": [asdict(row) for row in evaluations],
                "registrations": [asdict(row) for row in registrations],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
