from __future__ import annotations

import pandas as pd

from ceynex.models.agriculture.baseline import AnnualNaiveModel


def test_annual_naive_forecast_has_an_80_percent_interval() -> None:
    frame = pd.DataFrame({"period": range(2011, 2026), "value": range(100, 250, 10)})

    points = AnnualNaiveModel(sector="agriculture", item="tea", unit="kg").fit(frame).predict(3)

    assert [point["period"] for point in points] == ["2026", "2027", "2028"]
    assert all(point["lower"] <= point["point"] <= point["upper"] for point in points)
    assert points[2]["upper"] - points[2]["lower"] > points[0]["upper"] - points[0]["lower"]


def test_drift_baseline_extends_the_average_annual_change() -> None:
    frame = pd.DataFrame({"period": [2020, 2021, 2022, 2023, 2024], "value": [10, 12, 14, 16, 18]})

    point = AnnualNaiveModel(sector="agriculture", item="cinnamon", strategy="drift").fit(frame).predict(1)[0]

    assert point["point"] == 20.0
