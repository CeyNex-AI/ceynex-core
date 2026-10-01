"""Dated snapshot folders: which is newest, and which is recent enough to reuse."""

from datetime import date

from ceynex.data.connectors._snapshots import recent_enough

TODAY = date(2026, 10, 2)


def test_no_limit_reuses_anything():
    assert recent_enough("2020-01-01", None, TODAY)
    assert recent_enough("not-a-date", None, TODAY)


def test_a_pull_inside_the_limit_is_reused():
    assert recent_enough("2026-09-24", 25, TODAY)


def test_a_pull_at_or_past_the_limit_is_not():
    assert not recent_enough("2026-09-07", 25, TODAY)
    assert not recent_enough("2026-08-26", 25, TODAY)


def test_an_undated_folder_is_never_recent_under_a_limit():
    assert not recent_enough("latest", 25, TODAY)
