"""Regression checks for agriculture fold-level forecast-error reporting."""

from __future__ import annotations

from eval.agriculture_forecast_errors import _error_rows, summary
from eval.backtest import Fold


def test_error_rows_keep_each_held_out_period_and_interval_result() -> None:
    rows = _error_rows(
        "tea",
        "export_volume",
        "kg",
        [
            Fold(
                index=0,
                train_periods=[2018, 2019],
                test_periods=[2020],
                predicted=[90.0],
                actual=[100.0],
                lower=[80.0],
                upper=[110.0],
            )
        ],
    )

    assert rows[0].fold == 1
    assert rows[0].train_end_period == 2019
    assert rows[0].test_period == 2020
    assert rows[0].absolute_error == 10.0
    assert rows[0].absolute_percentage_error == 0.10
    assert rows[0].covered_by_80_interval is True


def test_zero_actual_is_not_given_an_invented_percentage_error() -> None:
    rows = _error_rows(
        "tea",
        "export_volume",
        "kg",
        [
            Fold(0, [2019], [2020], [3.0], [0.0], [-1.0], [4.0])
        ],
    )

    assert rows[0].absolute_percentage_error is None


def test_summary_reports_largest_error_and_coverage() -> None:
    rows = _error_rows(
        "cinnamon",
        "producer_price",
        "USD/kg",
        [
            Fold(0, [2020], [2021], [8.0], [10.0], [7.0], [9.0]),
            Fold(1, [2020, 2021], [2022], [10.0], [11.0], [9.0], [12.0]),
        ],
    )

    result = summary(rows)[0]

    assert result["held_out_observations"] == 2
    assert result["largest_absolute_error"] == 2.0
    assert result["largest_absolute_percentage_error"] == 0.2
    assert result["interval_coverage"] == 0.5
