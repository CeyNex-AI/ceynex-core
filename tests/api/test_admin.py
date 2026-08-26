"""Assertions for the admin routes (SRS 3.5.4): retrain, ingest triggers, and
DQ review, plus the role gate every one of them sits behind.

No real Postgres/model registry/network here — `ceynex.models.registry`,
`ceynex.data.reader.annual_series`, `ceynex.data.pipeline.run_source`, and
`ceynex.api.admin`'s DB functions are all monkeypatched at their real module
attribute, which works even though the routes import them lazily (a lazy
import still resolves the name from the same live module object monkeypatch
modified). `tests/api/test_admin_integration.py` proves the real-Postgres half
of `ceynex/api/admin.py`.
"""

from __future__ import annotations

import psycopg
import pytest
from fastapi.testclient import TestClient

from ceynex.api import admin as admin_module
from ceynex.api import deps as deps_module
from ceynex.api.auth import DemoUser, issue_token
from ceynex.api.deps import Runtime
from ceynex.api.main import app
from ceynex.llm import ProviderStatus
from ceynex.models.registry import ModelMetadata


@pytest.fixture
def client():
    return TestClient(app)


class FakeKGForRuntime:
    async def verify_connectivity(self):
        return True

    async def close(self) -> None:
        return None


class FakeLLMForStatus:
    """Just enough of LLMReasoningClient's surface for /api/admin/llm/status:
    the route only ever calls `.provider_status()`."""

    def __init__(self, status: dict[str, ProviderStatus]):
        self._status = status

    def provider_status(self) -> dict[str, ProviderStatus]:
        return self._status


@pytest.fixture
def llm_status_client(request):
    """A client whose runtime's LLM reports a specific provider_status()."""
    status = getattr(
        request,
        "param",
        {
            "openai": ProviderStatus(configured=True, status="ok", last_checked_at=1_700_000_000.0),
            "openrouter": ProviderStatus(configured=True, status="unknown"),
        },
    )
    deps_module.set_runtime(
        Runtime(kg=FakeKGForRuntime(), llm=FakeLLMForStatus(status), deps=None, graph=None)
    )
    try:
        yield TestClient(app)
    finally:
        deps_module.set_runtime(None)


def token_for(email: str, role: str) -> str:
    return issue_token(DemoUser(email=email, role=role, password_hash=b""))


def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {token_for('admin@ceynex.dev', 'admin')}"}


def researcher_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {token_for('researcher@ceynex.dev', 'researcher')}"}


ADMIN_ROUTES = [
    ("GET", "/api/admin/models", None),
    ("POST", "/api/admin/retrain", {"sector": "agriculture", "item": "cinnamon"}),
    ("POST", "/api/admin/pipeline/ingest", {}),
    ("GET", "/api/admin/pipeline/status", None),
    ("GET", "/api/admin/dq-flags", None),
    ("POST", "/api/admin/dq-flags/1/resolve", None),
    ("GET", "/api/admin/llm/status", None),
]


# --- role gate ---------------------------------------------------------


@pytest.mark.parametrize("method,path,body", ADMIN_ROUTES)
def test_every_admin_route_rejects_a_non_admin_role(client, method, path, body):
    response = client.request(method, path, json=body, headers=researcher_headers())
    assert response.status_code == 403


@pytest.mark.parametrize("method,path,body", ADMIN_ROUTES)
def test_every_admin_route_rejects_no_token(client, method, path, body):
    response = client.request(method, path, json=body)
    assert response.status_code == 401


# --- models / retrain ----------------------------------------------------


def test_list_models_returns_the_registry_contents(client, monkeypatch):
    meta = ModelMetadata(
        sector="agriculture", item="cinnamon", target="export_value_usd",
        version="v1", saved_at="2026-08-21T00:00:00+00:00",
        model_class="TimeSeriesModel", model_module="ceynex.models.timeseries",
        training_rows=10, metrics={"mape": 0.06}, interval_level=0.8, notes=None,
    )
    monkeypatch.setattr("ceynex.models.registry.list_models", lambda: [meta])

    response = client.get("/api/admin/models", headers=admin_headers())

    assert response.status_code == 200
    assert response.json()["models"] == [
        {
            "sector": "agriculture", "item": "cinnamon", "target": "export_value_usd",
            "version": "v1", "saved_at": "2026-08-21T00:00:00+00:00",
            "model_class": "TimeSeriesModel", "training_rows": 10,
            "metrics": {"mape": 0.06}, "interval_level": 0.8, "notes": None,
        }
    ]


def test_retrain_returns_the_new_version(client, monkeypatch):
    import pandas as pd

    frame = pd.DataFrame({"period": [2022, 2023], "value": [100.0, 110.0]})
    monkeypatch.setattr("ceynex.data.reader.annual_series", lambda item, sector, target: frame)

    retrained = ModelMetadata(
        sector="agriculture", item="cinnamon", target="export_value_usd",
        version="v2", saved_at="2026-08-21T01:00:00+00:00",
        model_class="TimeSeriesModel", model_module="ceynex.models.timeseries",
        training_rows=2, metrics=None, interval_level=0.8, notes="retrained",
    )
    monkeypatch.setattr(
        "ceynex.models.registry.retrain",
        lambda sector, item, target, df, **kw: retrained,
    )

    response = client.post(
        "/api/admin/retrain",
        json={"sector": "agriculture", "item": "cinnamon"},
        headers=admin_headers(),
    )

    assert response.status_code == 200
    assert response.json()["version"] == "v2"
    assert response.json()["training_rows"] == 2


def test_retrain_with_no_data_is_a_422(client, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "ceynex.data.reader.annual_series", lambda item, sector, target: pd.DataFrame()
    )

    response = client.post(
        "/api/admin/retrain",
        json={"sector": "agriculture", "item": "nothing-here"},
        headers=admin_headers(),
    )
    assert response.status_code == 422


def test_retrain_when_the_dataset_is_unreachable_is_a_503(client, monkeypatch):
    from ceynex.data.reader import DatasetUnavailableError

    def boom(item, sector, target):
        raise DatasetUnavailableError("db down")

    monkeypatch.setattr("ceynex.data.reader.annual_series", boom)

    response = client.post(
        "/api/admin/retrain",
        json={"sector": "agriculture", "item": "cinnamon"},
        headers=admin_headers(),
    )
    assert response.status_code == 503


def test_retrain_of_a_never_registered_item_is_a_404(client, monkeypatch):
    import pandas as pd

    from ceynex.models.registry import RegistryError

    monkeypatch.setattr(
        "ceynex.data.reader.annual_series",
        lambda item, sector, target: pd.DataFrame({"period": [2023], "value": [1.0]}),
    )

    def boom(sector, item, target, df, **kw):
        raise RegistryError("no versions registered")

    monkeypatch.setattr("ceynex.models.registry.retrain", boom)

    response = client.post(
        "/api/admin/retrain",
        json={"sector": "agriculture", "item": "never-trained"},
        headers=admin_headers(),
    )
    assert response.status_code == 404


# --- ingest ----------------------------------------------------------------


def test_ingest_with_an_unknown_source_is_a_422(client):
    response = client.post(
        "/api/admin/pipeline/ingest", json={"sources": ["not-a-real-source"]}, headers=admin_headers()
    )
    assert response.status_code == 422


def test_ingest_reports_a_result_per_source(client, monkeypatch):
    from ceynex.data.writer import WriteResult

    def fake_run_source(name, writer):
        return WriteResult(
            source_id=name, run_id=1, rows_in=10, rows_written=10, dq_flags=0,
            parquet_path=None, status="success",
        )

    monkeypatch.setattr("ceynex.data.pipeline.run_source", fake_run_source)

    response = client.post(
        "/api/admin/pipeline/ingest", json={"sources": ["edb"]}, headers=admin_headers()
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results == [
        {
            "source_id": "edb", "status": "success", "rows_in": 10, "rows_written": 10,
            "dq_flags": 0, "error": None, "warnings": [],
        }
    ]


def test_ingest_a_failing_source_does_not_fail_the_whole_request(client, monkeypatch):
    def boom(name, writer):
        raise RuntimeError("connector exploded")

    monkeypatch.setattr("ceynex.data.pipeline.run_source", boom)

    response = client.post(
        "/api/admin/pipeline/ingest", json={"sources": ["edb"]}, headers=admin_headers()
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "failed"
    assert "connector exploded" in result["error"]


# --- pipeline status ---------------------------------------------------


def test_pipeline_status_returns_recent_runs(client, monkeypatch):
    run = admin_module.PipelineRun(
        run_id=1, source_id="edb", started_at="2026-08-21T00:00:00+00:00",
        finished_at="2026-08-21T00:00:05+00:00", status="success", rows_written=100, error=None,
    )
    monkeypatch.setattr(admin_module, "pipeline_status", lambda: [run])

    response = client.get("/api/admin/pipeline/status", headers=admin_headers())

    assert response.status_code == 200
    assert response.json()["runs"][0]["source_id"] == "edb"


def test_pipeline_status_outage_is_a_503(client, monkeypatch):
    def boom():
        raise psycopg.OperationalError("db down")

    monkeypatch.setattr(admin_module, "pipeline_status", boom)

    response = client.get("/api/admin/pipeline/status", headers=admin_headers())
    assert response.status_code == 503


# --- dq review ---------------------------------------------------------


def test_dq_flags_lists_unresolved_first(client, monkeypatch):
    flag = admin_module.DQFlag(
        flag_id=1, item="cinnamon", hs_code="0906", partner_iso3="DEU",
        period_start="2023-01-01", metric="export_value_usd", source_a="EDB",
        value_a=100.0, source_b="JAAF", value_b=120.0, pct_diff=18.2,
        severity="material", detected_at="2026-08-21T00:00:00+00:00", resolved=False,
    )
    captured = {}

    def fake_list(*, resolved=None, severity=None, limit=50):
        captured["resolved"] = resolved
        captured["severity"] = severity
        return [flag]

    monkeypatch.setattr(admin_module, "list_dq_flags", fake_list)

    response = client.get(
        "/api/admin/dq-flags?resolved=false&severity=material", headers=admin_headers()
    )

    assert response.status_code == 200
    assert response.json()["flags"][0]["severity"] == "material"
    assert captured == {"resolved": False, "severity": "material"}


def test_dq_flags_outage_is_a_503(client, monkeypatch):
    def boom(*, resolved=None, severity=None, limit=50):
        raise psycopg.OperationalError("db down")

    monkeypatch.setattr(admin_module, "list_dq_flags", boom)

    response = client.get("/api/admin/dq-flags", headers=admin_headers())
    assert response.status_code == 503


def test_resolving_a_flag_that_exists_returns_true(client, monkeypatch):
    monkeypatch.setattr(admin_module, "resolve_dq_flag", lambda flag_id: True)

    response = client.post("/api/admin/dq-flags/1/resolve", headers=admin_headers())

    assert response.status_code == 200
    assert response.json() == {"flag_id": 1, "resolved": True}


def test_resolving_a_flag_that_does_not_exist_is_a_404(client, monkeypatch):
    monkeypatch.setattr(admin_module, "resolve_dq_flag", lambda flag_id: False)

    response = client.post("/api/admin/dq-flags/999/resolve", headers=admin_headers())
    assert response.status_code == 404


# --- LLM provider status (SRS 3.5.4) --------------------------------------


def test_llm_status_reports_both_providers(llm_status_client):
    response = llm_status_client.get("/api/admin/llm/status", headers=admin_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["openai"]["status"] == "ok"
    assert body["openai"]["configured"] is True
    assert body["openrouter"]["status"] == "unknown"


def test_llm_status_converts_epoch_seconds_to_iso8601(llm_status_client):
    response = llm_status_client.get("/api/admin/llm/status", headers=admin_headers())

    # 1_700_000_000 is 2023-11-14T22:13:20+00:00 -- exact instant matters less
    # than proving it's a real ISO timestamp, not the raw epoch float.
    assert response.json()["openai"]["last_checked_at"] == "2023-11-14T22:13:20+00:00"


@pytest.mark.parametrize(
    "llm_status_client",
    [{"openai": ProviderStatus(configured=False, status="not_configured"),
      "openrouter": ProviderStatus(configured=False, status="not_configured")}],
    indirect=True,
)
def test_llm_status_reports_not_configured_with_no_timestamp(llm_status_client):
    response = llm_status_client.get("/api/admin/llm/status", headers=admin_headers())

    body = response.json()
    assert body["openai"] == {
        "configured": False, "status": "not_configured", "last_error": None, "last_checked_at": None,
    }


@pytest.mark.parametrize(
    "llm_status_client",
    [{"openai": ProviderStatus(configured=True, status="down", last_error="rate limited", last_checked_at=1_700_000_000.0),
      "openrouter": ProviderStatus(configured=True, status="ok", last_checked_at=1_700_000_001.0)}],
    indirect=True,
)
def test_llm_status_distinguishes_a_down_openai_from_an_ok_openrouter(llm_status_client):
    """The dashboard's whole point: an admin sees GPT-4o needs attention
    while the free failsafe is covering it in the meantime."""
    response = llm_status_client.get("/api/admin/llm/status", headers=admin_headers())

    body = response.json()
    assert body["openai"]["status"] == "down"
    assert body["openai"]["last_error"] == "rate limited"
    assert body["openrouter"]["status"] == "ok"
