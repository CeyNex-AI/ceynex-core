"""Assertions for the shared rolling-origin harness (SRS 3.1.10).

This file exists mainly to prove one thing: the harness cannot see the future.
Everything else in the evaluation report rests on that, and a leak here would
make three members' numbers wrong in the same invisible direction.
"""

import math

import pandas as pd
import pytest

from ceynex.models.timeseries import TimeSeriesModel
from eval.backtest import (
    MIN_TRAIN,
    BacktestError,
    Fold,
    _folds,
    coverage,
    mape,
    rmse,
    rolling_origin,
    rolling_origin_folds,
    summarize,
)

TREND = pd.DataFrame(
    {"period": list(range(2015, 2025)), "value": [100, 112, 121, 133, 129, 145, 158, 166, 181, 190]}
)


def fitted():
    return TimeSeriesModel(sector="agriculture", item="cinnamon").fit(TREND)


# --- the leakage guarantee -----------------------------------------------


def test_no_fold_trains_on_a_period_it_later_predicts():
    """The property that makes this a backtest rather than a fitted curve."""
    for fold in rolling_origin_folds(fitted(), TREND, folds=3):
        assert max(fold.train_periods) < min(fold.test_periods), (
            f"fold {fold.index} trained through {max(fold.train_periods)} "
            f"and then predicted {fold.test_periods}"
        )


def test_the_training_window_expands_rather_than_slides():
    folds = rolling_origin_folds(fitted(), TREND, folds=3)
    sizes = [len(f.train_periods) for f in folds]
    assert sizes == sorted(sizes), "training windows shrank; this is not an expanding origin"
    assert len(set(sizes)) > 1, "every fold trained on the same data"


def test_every_split_respects_the_minimum_training_window():
    for train_end, _ in _folds(20, folds=5, horizon=1):
        assert train_end >= MIN_TRAIN


def test_a_series_too_short_for_the_requested_folds_is_refused():
    tiny = pd.DataFrame({"period": [2021, 2022, 2023, 2024], "value": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(BacktestError):
        rolling_origin(fitted(), tiny, folds=3)


def test_the_passed_model_is_not_the_one_that_gets_fitted():
    """A model already fitted on the whole series would leak it into every fold."""
    model = fitted()
    before = model.predict(1)[0]["point"]
    rolling_origin(model, TREND, folds=3)
    assert model.predict(1)[0]["point"] == before, "the harness refitted the caller's model"


# --- the metrics ---------------------------------------------------------


def test_mape_is_a_fraction_not_a_percentage():
    assert mape([100.0, 200.0], [110.0, 180.0]) == pytest.approx(0.10)


def test_mape_skips_zero_actuals_rather_than_inventing_a_denominator():
    """Adding epsilon turns a rounding constant into an enormous error term."""
    assert mape([0.0, 100.0], [5.0, 110.0]) == pytest.approx(0.10)


def test_mape_is_nan_when_every_actual_is_zero():
    assert math.isnan(mape([0.0, 0.0], [1.0, 2.0]))


def test_rmse_punishes_one_large_miss_harder_than_mape_does():
    spread = rmse([100.0, 100.0], [90.0, 110.0])
    concentrated = rmse([100.0, 100.0], [100.0, 120.0])
    assert concentrated > spread


def test_coverage_counts_actuals_inside_the_band():
    assert coverage([5.0, 5.0, 5.0], [0.0, 6.0, 4.0], [10.0, 9.0, 6.0]) == pytest.approx(2 / 3)


def test_coverage_counts_a_value_exactly_on_the_bound_as_covered():
    assert coverage([10.0], [10.0], [20.0]) == 1.0


def test_summarize_pools_folds_rather_than_averaging_their_averages():
    """Unequal fold sizes make an average-of-averages quietly wrong."""
    folds = [
        Fold(0, [2020], [2021], predicted=[110.0], actual=[100.0], lower=[90.0], upper=[120.0]),
        Fold(1, [2020, 2021], [2022, 2023], predicted=[100.0, 100.0], actual=[100.0, 100.0], lower=[0.0, 0.0], upper=[0.0, 0.0]),
    ]
    result = summarize(folds)
    assert result["observations"] == 3.0
    assert result["mape"] == pytest.approx(0.10 / 3)


def test_metrics_report_how_many_observations_they_rest_on():
    """A MAPE from two points is not the same claim as a MAPE from fifty."""
    result = rolling_origin(fitted(), TREND, folds=3)
    assert result["observations"] > 0
    assert result["folds"] > 0
