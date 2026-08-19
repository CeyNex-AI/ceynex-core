"""Assertions for SRS 3.1.10 — the shared model registry.

The registry is written once and used by three people, so the tests that matter
are the ones about *not losing someone else's model*: version collisions,
overwrites, and the lookup returning the newest rather than the first found.
"""

import pandas as pd
import pytest

from ceynex.models import registry
from ceynex.models.registry import RegistryError
from ceynex.models.timeseries import TimeSeriesModel

SERIES = pd.DataFrame(
    {"period": list(range(2015, 2025)), "value": [100, 112, 121, 133, 129, 145, 158, 166, 181, 190]}
)


def fitted(item="cinnamon", sector="agriculture"):
    return TimeSeriesModel(sector=sector, item=item).fit(SERIES)


def test_save_then_load_round_trips_a_working_model():
    saved = registry.save(fitted(), training_rows=len(SERIES))
    loaded = registry.load(saved.sector, saved.item, saved.target, saved.version)

    assert loaded.predict(1), "a model that cannot predict after a round trip is not saved"
    assert loaded.item == "cinnamon"


def test_two_saves_in_the_same_second_do_not_overwrite_each_other():
    """Version stamps are second-granular; a loop over model families collides."""
    first = registry.save(fitted())
    second = registry.save(fitted())

    assert first.version != second.version
    assert len(registry.list_models()) == 2


def test_list_models_does_not_count_a_version_twice_through_the_latest_link():
    registry.save(fitted())
    assert len(registry.list_models()) == 1


def test_load_latest_returns_the_most_recently_saved():
    registry.save(fitted(), version="v1")
    newest = registry.save(fitted(), version="v2")

    loaded = registry.load_latest(item="cinnamon")
    assert loaded is not None
    assert loaded.version == newest.version


def test_load_latest_is_none_when_nothing_is_registered():
    """The normal state for most of the sprint. Must not raise."""
    assert registry.load_latest(item="nobody-has-trained-this") is None


def test_load_latest_matches_items_case_and_spacing_insensitively():
    registry.save(TimeSeriesModel(sector="apparel", item="Apparel Knit").fit(SERIES))
    assert registry.load_latest(item="apparel_knit") is not None


def test_load_latest_can_be_narrowed_by_sector():
    registry.save(fitted(item="tea", sector="agriculture"))
    assert registry.load_latest(item="tea", sector="apparel") is None
    assert registry.load_latest(item="tea", sector="agriculture") is not None


def test_a_model_that_does_not_say_what_it_forecasts_is_rejected():
    model = fitted()
    model.item = ""
    with pytest.raises(RegistryError, match="item"):
        registry.save(model)


def test_metrics_can_be_attached_after_the_model_is_saved():
    """Fitting and backtesting are separate steps, so metrics arrive later."""
    saved = registry.save(fitted())
    assert saved.metrics is None

    updated = registry.record_metrics(saved, {"mape": 0.1, "rmse": 5.0, "coverage": 0.8})
    assert updated.metrics["mape"] == 0.1
    assert registry.list_models()[0].metrics["mape"] == 0.1


def test_model_id_names_the_version():
    """Evidence saying 'a model produced this' without saying which is not a trail."""
    saved = registry.save(fitted(), version="v7")
    assert saved.model_id.endswith("@v7")


def test_an_empty_registry_lists_nothing_rather_than_failing():
    assert registry.list_models() == []


def test_loading_an_unregistered_model_raises():
    with pytest.raises(RegistryError):
        registry.load("agriculture", "nothing", "export_value_usd")


def test_retrain_keeps_the_model_class_and_leaves_the_old_version_in_place():
    original = registry.save(fitted())
    retrained = registry.retrain("agriculture", "cinnamon", "export_value_usd", SERIES)

    assert retrained.model_class == original.model_class, "a retrain must not swap the model family"
    assert retrained.version != original.version
    assert len(registry.list_models()) == 2, "the previous version must stay loadable"
