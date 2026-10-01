"""The drift / damped-ETS / Theta combination (ceynex/models/combination.py)."""

import pickle

import pandas as pd
import pytest

from ceynex.models.combination import CombinationModel, _yearly_steps
from eval.backtest import _build, rolling_origin

# Live tea export value, USD millions, with 2018 missing at source as in Comtrade.
TEA = pd.DataFrame({
    "period": [2015, 2016, 2017, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    "value": [1310, 1241, 1495, 1304, 1307, 1368, 1283, 1271, 1373, 1432],
})


def model(**kwargs):
    return CombinationModel(sector="agriculture", item="tea", **kwargs)


def test_it_forecasts_with_an_interval_around_the_point():
    points = model().fit(TEA).predict(2)

    assert [p["period"] for p in points] == ["2026", "2027"]
    for p in points:
        assert p["lower"] <= p["point"] <= p["upper"]
    assert points[1]["upper"] - points[1]["lower"] > points[0]["upper"] - points[0]["lower"], "uncertainty grows with horizon"


def test_all_three_members_fit_on_ten_annual_points():
    fitted = model().fit(TEA)
    assert fitted.fitted_members == ("drift", "damped_ets", "theta")
    assert fitted.fitted_family == "Combination(drift, damped_ets, theta)"


def test_a_missing_year_is_not_treated_as_one_step():
    """2017 -> 2019 is two years of change, so drift spreads it over both."""
    steps = _yearly_steps([2017, 2019, 2020], [100.0, 120.0, 125.0])
    assert steps == [10.0, 10.0, 5.0]


def test_the_shared_harness_can_backtest_and_clone_it():
    """rolling_origin rebuilds the model from describe_params for every fold."""
    metrics = rolling_origin(model(), TEA, folds=3, horizon=1)
    assert 0 < metrics["mape"] < 0.2
    assert 0.0 <= metrics["coverage"] <= 1.0


def test_it_survives_the_registry_pickle_round_trip():
    fitted = model().fit(TEA)
    restored = pickle.loads(pickle.dumps(fitted))
    assert restored.predict(1)[0]["point"] == pytest.approx(fitted.predict(1)[0]["point"])


def test_unknown_members_are_refused():
    with pytest.raises(ValueError, match="unknown combination members"):
        model(members=("drift", "prophet"))


def test_the_backtest_cli_can_build_it():
    built = _build("combination", sector="agriculture", item="tea", target="export_value_usd")
    assert isinstance(built, CombinationModel)
