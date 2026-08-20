import math

import pandas as pd
import pytest

from ceynex.models.apparel import NaiveApparelForecastModel


def _usa_df():
    """Real EDB Apparel sub-category exports to the USA, 2020-2024 (deduplicated,
    validated against the source PDFs earlier this session). Not a synthetic
    fixture: this is the exact series the agent's own Cypher (`_PARTNER_QUERY`,
    LIMIT 5) would hand the model for a real "apparel exports to the US" query.
    """
    return pd.DataFrame(
        {
            "period": [2020, 2021, 2022, 2023, 2024],
            "value": [
                1_649_270_000.0,
                2_082_600_000.0,
                2_300_240_000.0,
                1_782_720_000.0,
                1_875_850_000.0,
            ],
        }
    )


def test_predict_requires_fit_first():
    model = NaiveApparelForecastModel("USA")
    with pytest.raises(RuntimeError):
        model.predict(1)


def test_predict_is_flat_carry_forward_of_last_observed_value():
    model = NaiveApparelForecastModel("USA").fit(_usa_df())
    points = model.predict(2)

    assert [p["period"] for p in points] == ["2025", "2026"]
    # Naive means every horizon step is the same point estimate -- the last
    # observed year (2024), not a fitted trend.
    assert points[0]["point"] == points[1]["point"] == pytest.approx(1_875_850_000.0)
    assert all(p["unit"] == "USD" for p in points)


def test_predict_interval_contains_point_and_widens_with_horizon():
    model = NaiveApparelForecastModel("USA").fit(_usa_df())
    points = model.predict(2)

    for p in points:
        assert p["lower"] <= p["point"] <= p["upper"]
        assert p["lower"] >= 0.0  # export values can't be negative

    # Uncertainty compounds over a longer horizon under the residual bootstrap.
    width_h1 = points[0]["upper"] - points[0]["lower"]
    width_h2 = points[1]["upper"] - points[1]["lower"]
    assert width_h2 > width_h1


def test_backtest_on_real_data_reports_honest_metrics():
    model = NaiveApparelForecastModel("USA").fit(_usa_df())
    metrics = model.backtest(folds=3)

    assert metrics["folds"] == 3.0
    assert metrics["n_obs"] == 5.0
    assert metrics["mape"] == pytest.approx(0.14485367079032996, rel=1e-6)
    assert metrics["rmse"] == pytest.approx(328566237.30992204, rel=1e-6)
    assert 0.0 <= metrics["coverage"] <= 1.0


def test_backtest_requires_fit_first():
    model = NaiveApparelForecastModel("USA")
    with pytest.raises(RuntimeError):
        model.backtest()


def test_backtest_clamps_folds_to_available_history():
    two_point_df = pd.DataFrame({"period": [2023, 2024], "value": [100.0, 110.0]})
    model = NaiveApparelForecastModel("USA").fit(two_point_df)

    # Only one possible rolling-origin split with 2 points, even though
    # folds=3 was requested.
    metrics = model.backtest(folds=3)
    assert metrics["folds"] == 1.0
    assert metrics["n_obs"] == 2.0


def test_backtest_on_single_observation_is_not_evaluable():
    one_point_df = pd.DataFrame({"period": [2024], "value": [100.0]})
    model = NaiveApparelForecastModel("USA").fit(one_point_df)

    metrics = model.backtest(folds=3)
    assert metrics["folds"] == 0.0
    assert math.isnan(metrics["mape"])
    assert math.isnan(metrics["rmse"])
    assert math.isnan(metrics["coverage"])


def test_wide_fallback_interval_when_fewer_than_two_residuals():
    # A single one-step residual (2 points): not enough to bootstrap from, so
    # predict() must still return a valid, required interval (contracts/
    # forecast.py makes lower/upper mandatory) via the documented fallback.
    two_point_df = pd.DataFrame({"period": [2023, 2024], "value": [100.0, 120.0]})
    model = NaiveApparelForecastModel("USA").fit(two_point_df)

    points = model.predict(1)
    p = points[0]
    assert p["point"] == pytest.approx(120.0)
    assert p["lower"] == pytest.approx(90.0)  # 120 * 0.75
    assert p["upper"] == pytest.approx(150.0)  # 120 * 1.25
