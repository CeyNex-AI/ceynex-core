"""Compare the registered forecast families on the annual export-value series.

    python -m eval.model_comparison                 # reads the dataset
    python -m eval.model_comparison --series s.json # offline: {"tea": {"period": [...], "value": [...]}, ...}

Scores SARIMA(1,1,0), LightGBM and the drift/damped-ETS/Theta combination with
the shared rolling-origin harness at 3 and 5 folds, then tests each family
against SARIMA on the individual one-year-ahead errors (Wilcoxon signed-rank).
This is the evidence behind `ceynex/models/combination.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings

import pandas as pd

from eval.backtest import rolling_origin, rolling_origin_folds

ITEMS = {
    "tea": "agriculture", "cinnamon": "agriculture", "rubber": "agriculture", "coconut": "agriculture",
    "apparel_knit": "apparel", "apparel_woven": "apparel",
}


def _families():
    from ceynex.models.combination import CombinationModel
    from ceynex.models.gbm import GradientBoostedModel
    from ceynex.models.timeseries import TimeSeriesModel

    return {"SARIMA(1,1,0)": TimeSeriesModel, "LightGBM": GradientBoostedModel, "Combination": CombinationModel}


def _load(path: str | None) -> dict[str, pd.DataFrame]:
    if path:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        return {item: pd.DataFrame({"period": [int(p) for p in d["period"]], "value": d["value"]})
                for item, d in raw.items()}
    from ceynex.data.reader import annual_series

    return {item: annual_series(item, sector=sector, target="export_value_usd") for item, sector in ITEMS.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--series", help="JSON file of series instead of the dataset")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    series, families = _load(args.series), _families()
    for folds in (3, 5):
        print(f"\nMAPE % (80% coverage) at {folds} folds")
        means = {}
        for name, cls in families.items():
            cells, mapes = [], []
            for item, frame in series.items():
                r = rolling_origin(cls(sector=ITEMS[item], item=item), frame, folds=folds, horizon=1)
                cells.append(f"{item} {r['mape'] * 100:.1f} ({r['coverage']:.2f})")
                mapes.append(r["mape"])
            means[name] = sum(mapes) / len(mapes)
            print(f"  {name:14} mean {means[name] * 100:5.2f}  | " + "  ".join(cells))

    from scipy.stats import wilcoxon

    def errors(cls) -> list[float]:
        out: list[float] = []
        for item, frame in series.items():
            for fold in rolling_origin_folds(cls(sector=ITEMS[item], item=item), frame, folds=5, horizon=1):
                out += [abs((a - p) / a) for a, p in zip(fold.actual, fold.predicted, strict=True)]
        return out

    base = errors(families["SARIMA(1,1,0)"])
    print("\nAgainst SARIMA, 5 folds, per one-year-ahead forecast:")
    for name in ("LightGBM", "Combination"):
        other = errors(families[name])
        better = sum(o < b for o, b in zip(other, base, strict=True))
        print(f"  {name:12} better in {better}/{len(base)}, Wilcoxon p = {wilcoxon(other, base).pvalue:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
