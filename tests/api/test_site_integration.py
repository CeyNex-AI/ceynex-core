"""Real-Postgres assertions for `ceynex/api/site_settings.py`: `ensure_table`/
`get_theme`/`set_theme` exercised against a live database instead of
monkeypatched. Same spirit as `tests/api/test_history_integration.py`.
"""

from __future__ import annotations

import psycopg
import pytest

from ceynex.api import site_settings
from ceynex.settings import postgres_dsn

TEST_ACTOR = "pytest-site@ceynex.dev"


@pytest.fixture
def clean_site_settings():
    """Resets the singleton row to the default rather than deleting it --
    `site_settings` always has exactly one row once `ensure_table` has run
    anywhere, and other tests/processes read it."""

    def reset():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM site_settings")
            conn.commit()

    site_settings.ensure_table()
    reset()
    yield
    reset()


@pytest.mark.integration
def test_get_theme_defaults_to_classic_with_no_row(clean_site_settings):
    assert site_settings.get_theme() == "classic"


@pytest.mark.integration
def test_set_theme_then_get_theme_reflects_it(clean_site_settings):
    assert site_settings.set_theme("signal-deck", updated_by=TEST_ACTOR) == "signal-deck"
    assert site_settings.get_theme() == "signal-deck"


@pytest.mark.integration
def test_set_theme_twice_updates_the_same_singleton_row(clean_site_settings):
    site_settings.set_theme("signal-deck", updated_by=TEST_ACTOR)
    site_settings.set_theme("classic", updated_by=TEST_ACTOR)

    assert site_settings.get_theme() == "classic"
    with psycopg.connect(postgres_dsn()) as conn:
        count = conn.execute("SELECT count(*) FROM site_settings").fetchone()[0]
    assert count == 1


@pytest.mark.integration
def test_set_theme_rejects_an_unknown_name(clean_site_settings):
    with pytest.raises(ValueError):
        site_settings.set_theme("neon", updated_by=TEST_ACTOR)
