"""Every registry test writes to a temp directory, never the real `models/`."""

import pytest


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("CEYNEX_MODELS_DIR", str(tmp_path / "models"))
    yield
