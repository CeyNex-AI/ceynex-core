"""Assertions for SRS 3.1.3 and 3.1.10 — the two concrete model families.

The contract's hard requirement is that a forecast is never a bare point, so
most of this file is about the interval: that it exists, contains its own point,
widens with the horizon, and does not claim a negative export value.
"""

import pandas as pd
import pytest

from ceynex.models.base import InsufficientHistoryError, NotFittedError
from ceynex.models.gbm import GradientBoostedModel
from ceynex.models.timeseries import TimeSeriesModel

TREND = pd.DataFrame(
    {"period": list(range(2015, 2025)), "value": [100, 112, 121, 133, 129, 145, 158, 166, 181, 190]}
)

FAMILIES = [TimeSeriesModel, GradientBoostedModel]


def fit(cls, frame=TREND):
    return cls(sector="agriculture", item="cinnamon").fit(frame)


@pytest.mark.parametrize("cls", FAMILIES)
def test_every_forecast_carries_an_interval_around_its_point(cls):
    """SRS 3.1.3. A point without a band is the thing the contract forbids."""
    for point in fit(cls).predict(3):
        assert point["lower"] <= point["point"] <= point["upper"]


@pytest.mark.parametrize("cls", FAMILIES)
def test_the_interval_widens_with_the_horizon(cls):
    """Independent shocks accumulate; a flat band understates far-out uncertainty."""
    points = fit(cls).predict(3)
    widths = [p["upper"] - p["lower"] for p in points]
    assert widths[-1] > widths[0], f"{cls.__name__} claims year 3 is as certain as year 1"


@pytest.mark.parametrize("cls", FAMILIES)
def test_the_lower_bound_is_never_negative(cls):
    """Export value cannot go below zero, so a band that does is not informative."""
    falling = pd.DataFrame({"period": list(range(2015, 2025)), "value": [900, 800, 700, 600, 500, 400, 300, 200, 100, 50]})
    for point in fit(cls, falling).predict(5):
        assert point["lower"] >= 0.0


@pytest.mark.parametrize("cls", FAMILIES)
def test_predicting_before_fitting_raises(cls):
    with pytest.raises(NotFittedError):
        cls(sector="agriculture", item="cinnamon").predict(1)


@pytest.mark.parametrize("cls", FAMILIES)
def test_a_series_too_short_to_model_is_refused_rather_than_fitted(cls):
    short = pd.DataFrame({"period": [2022, 2023, 2024], "value": [1.0, 2.0, 3.0]})
    with pytest.raises(InsufficientHistoryError):
        fit(cls, short)


@pytest.mark.parametrize("cls", FAMILIES)
def test_period_and_value_columns_are_discovered_not_mandated(cls):
    """Three members build frames from three sources; rigid names help nobody."""
    renamed = TREND.rename(columns={"period": "year", "value": "export_value_usd"})
    assert fit(cls, renamed).predict(1)


@pytest.mark.parametrize("cls", FAMILIES)
def test_partner_rows_for_one_year_are_summed_into_one_observation(cls):
    """A frame with several partners per year is one series, not duplicates."""
    doubled = pd.concat([TREND.assign(value=TREND["value"] / 2)] * 2, ignore_index=True)
    model = fit(cls, doubled)
    assert model.predict(1)[0]["point"] == pytest.approx(fit(cls).predict(1)[0]["point"], rel=0.05)


# --- the regression test that matters ------------------------------------


def test_the_boosted_model_can_forecast_above_its_training_range():
    """Trees cannot extrapolate, so this model fits year-on-year changes.

    Fitted on levels it returned a flat line on a trending series — MAPE 0.12,
    interval coverage 0.00. This test fails if anyone changes it back.
    """
    model = fit(GradientBoostedModel)
    points = model.predict(3)
    highest_seen = TREND["value"].max()

    assert points[-1]["point"] > highest_seen, (
        "the boosted model predicted no higher than its training data on a "
        "rising series — it is fitting levels again, not changes"
    )
    assert points[2]["point"] > points[0]["point"], "the trend is not being carried forward"


def test_the_time_series_model_falls_back_when_sarima_will_not_converge():
    model = fit(TimeSeriesModel)
    assert model.fitted_family is not None
    assert model.fitted_family.startswith(("SARIMA", "ETS"))


@pytest.mark.parametrize("cls", FAMILIES)
def test_describe_params_is_enough_to_rebuild_the_model(cls):
    """The registry clones models from this dict; a gap breaks retrain and backtest."""
    model = fit(cls)
    rebuilt = type(model)(**model.describe_params())
    assert rebuilt.item == model.item
