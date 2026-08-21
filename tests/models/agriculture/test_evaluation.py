from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ceynex.models import registry
from ceynex.models.agriculture import evaluation


def _write_sources(root: Path) -> None:
    tea = root / "tea_board" / "2026-08-15" / "tea_annual_production_exports_2011_2025.xlsx"
    tea.parent.mkdir(parents=True)
    exports = pd.DataFrame({"year": range(2011, 2026), "total_exports_mt": range(200, 350, 10)})
    with pd.ExcelWriter(tea, engine="openpyxl") as writer:
        exports.to_excel(writer, sheet_name="Exports", index=False, startrow=3)

    faostat = root / "faostat" / "2026-08-15"
    faostat.mkdir(parents=True)
    prices = pd.DataFrame(
        {
            "Area": ["Sri Lanka"] * 34,
            "Area Code (M49)": [144] * 34,
            "Item": ["Cinnamon and cinnamon-tree flowers, raw"] * 34,
            "Element": ["Producer Price (USD/tonne)"] * 34,
            "Year": range(1991, 2025),
            "Value": range(1000, 35000, 1000),
        }
    )
    prices.to_csv(faostat / "FAOSTAT producer prices.csv", index=False)


def test_sufficiency_report_records_annual_short_series(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    report = evaluation.sufficiency_report(tmp_path)

    assert [(row.item, row.observations, row.period_start, row.period_end) for row in report] == [
        ("tea", 15, 2011, 2025),
        ("cinnamon", 34, 1991, 2024),
    ]
    assert all(row.frequency == "A" and row.below_complex_model_threshold for row in report)


def test_gbm_requires_a_material_gain_over_simple_models() -> None:
    simple = [
        evaluation.ModelEvaluation("cinnamon", "annual_naive", {"mape": 0.10}, False),
        evaluation.ModelEvaluation("cinnamon", "sarima_or_ets", {"mape": 0.09}, False),
    ]
    not_enough = evaluation.ModelEvaluation("cinnamon", "gbm", {"mape": 0.086}, False)
    material = evaluation.ModelEvaluation("cinnamon", "gbm", {"mape": 0.080}, False)

    assert evaluation._should_select_gbm(not_enough, simple) is False
    assert evaluation._should_select_gbm(material, simple) is True


def test_selected_models_register_with_complete_reproducibility_metadata(tmp_path: Path, monkeypatch) -> None:
    _write_sources(tmp_path)
    selected = [
        evaluation.ModelEvaluation("tea", "annual_naive", {"mape": 0.03, "rmse": 8.5, "coverage": 1.0, "folds": 3.0}, True),
        evaluation.ModelEvaluation("cinnamon", "annual_naive", {"mape": 0.12, "rmse": 1.2, "coverage": 1 / 3, "folds": 3.0}, True),
    ]
    monkeypatch.setattr(evaluation, "evaluate_agriculture_models", lambda _root: selected)

    registered = evaluation.register_selected_models(tmp_path, git_sha="b" * 40)

    assert [(model.item, model.target, model.model_class) for model in registered] == [
        ("tea", "export_volume", "AnnualNaiveModel"),
        ("cinnamon", "producer_price", "AnnualNaiveModel"),
    ]
    cinnamon, tea = sorted(registry.list_models(), key=lambda model: model.item)
    assert cinnamon.training_rows == 34
    assert cinnamon.training_window == {"period_start": 1991, "period_end": 2024}
    assert cinnamon.source.startswith("FAOSTAT")
    assert tea.training_rows == 15
    assert tea.training_window == {"period_start": 2011, "period_end": 2025}
    assert "Liyanage/Silva" in cinnamon.notes
    assert all(model.interval_level == 0.80 and model.git_sha == "b" * 40 for model in registered)


def test_registration_rejects_a_non_commit_git_sha(tmp_path: Path, monkeypatch) -> None:
    _write_sources(tmp_path)
    selected = [
        evaluation.ModelEvaluation("tea", "annual_naive", {"mape": 0.03, "rmse": 8.5, "coverage": 1.0, "folds": 3.0}, True),
        evaluation.ModelEvaluation("cinnamon", "annual_naive", {"mape": 0.12, "rmse": 1.2, "coverage": 1 / 3, "folds": 3.0}, True),
    ]
    monkeypatch.setattr(evaluation, "evaluate_agriculture_models", lambda _root: selected)

    with pytest.raises(ValueError, match="40-character"):
        evaluation.register_selected_models(tmp_path, git_sha="not-a-commit")
