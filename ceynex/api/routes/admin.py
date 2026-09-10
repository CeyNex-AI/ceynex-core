"""Admin routes — SRS 3.5.4. Retrain, ingest triggers and DQ review are the
three named items; all three exist elsewhere as real CLI hooks
(`ceynex.models.registry.retrain`, `ceynex.data.pipeline.run_source`,
`ceynex/api/admin.py`'s `dq_flag` reads) so these routes are thin wrappers,
not new logic — see `docs/DEFERRED.md`'s note on why that was left for M3
rather than guessed at by whoever seeded `ceynex/api/`.

Every route requires `require_admin` (`ceynex/api/routes/auth.py`) — a
signed-in researcher/exporter/policymaker gets 403, not just a hidden nav
link, since a retrain or ingest run is a real, if reversible, side effect.

Retraining and ingestion are both blocking (model fitting, network I/O) and
have no place on `/api/query`'s SRS 3.4.1 latency budget — they run via
`asyncio.to_thread` so the event loop keeps serving other requests, but the
HTTP response itself still waits for the real work to finish rather than
returning a job id to poll. That is a deliberate simplification for a demo
scale of "a handful of sources, a handful of years of annual data" — a queue
would be the right shape at a size where either operation takes minutes, not
seconds.
"""

from __future__ import annotations

import asyncio
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from ceynex.api import admin, audit, users
from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.routes.auth import TokenPayload, require_admin
from ceynex.api.schemas import (
    AuditLogItem,
    AuditLogResponse,
    CreateUserRequest,
    DQFlagItem,
    DQFlagsResponse,
    IngestRequest,
    IngestResponse,
    IngestResultItem,
    LLMStatusResponse,
    ModelsResponse,
    ModelSummary,
    PipelineRunItem,
    PipelineStatusResponse,
    ProviderStatusItem,
    ResolveDQFlagResponse,
    RetrainRequest,
    SetRoleRequest,
    UserAdminItem,
    UserMutationResponse,
    UsersResponse,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


async def _audit(admin_user: TokenPayload, action: str, target: str | None) -> None:
    """Write the audit row *before* the mutation it covers — see
    `ceynex/api/audit.py`'s module docstring for why a failed write must block
    the action rather than let it run unlogged."""
    try:
        await asyncio.to_thread(
            audit.record, actor_email=admin_user.email, action=action, target=target
        )
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=503, detail="could not write audit log; action not performed"
        ) from exc


# --- LLM provider status -------------------------------------------------


def _iso(epoch_s: float | None) -> str | None:
    if epoch_s is None:
        return None
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch_s, tz=UTC).isoformat()


@router.get("/llm/status", response_model=LLMStatusResponse)
async def llm_status(
    # require_admin first: FastAPI resolves dependencies in parameter order,
    # and a bad/missing token should 401/403 even if the runtime dependency
    # below would itself fail (e.g. in a test with no Runtime configured).
    _admin: TokenPayload = Depends(require_admin),  # noqa: B008
    runtime: Runtime = Depends(get_runtime),  # noqa: B008 - FastAPI's dependency idiom
) -> LLMStatusResponse:
    status = runtime.llm.provider_status()
    return LLMStatusResponse(
        openai=ProviderStatusItem(
            configured=status["openai"].configured,
            status=status["openai"].status,
            last_error=status["openai"].last_error,
            last_checked_at=_iso(status["openai"].last_checked_at),
        ),
        openrouter=ProviderStatusItem(
            configured=status["openrouter"].configured,
            status=status["openrouter"].status,
            last_error=status["openrouter"].last_error,
            last_checked_at=_iso(status["openrouter"].last_checked_at),
        ),
    )


# --- models / retrain --------------------------------------------------


def _model_summary(m: Any) -> ModelSummary:  # ceynex.models.registry.ModelMetadata
    return ModelSummary(
        sector=m.sector,
        item=m.item,
        target=m.target,
        version=m.version,
        saved_at=m.saved_at,
        model_class=m.model_class,
        training_rows=m.training_rows,
        metrics=m.metrics,
        interval_level=m.interval_level,
        notes=m.notes,
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(_admin: TokenPayload = Depends(require_admin)) -> ModelsResponse:  # noqa: B008
    # Imported lazily -- pulls in the full statsmodels/lightgbm/scikit-learn
    # stack, which has no business loading just to serve a login or a query.
    from ceynex.models import registry

    models = await asyncio.to_thread(registry.list_models)
    return ModelsResponse(models=[_model_summary(m) for m in models])


def _do_retrain(sector: str, item: str, target: str) -> Any:
    from ceynex.data.reader import DatasetUnavailableError, annual_series
    from ceynex.models import registry

    try:
        frame = annual_series(item, sector=sector, target=target)
    except DatasetUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if frame.empty:
        raise HTTPException(
            status_code=422, detail=f"no annual {target} rows for {sector}/{item}"
        )
    try:
        return registry.retrain(sector, item, target, frame)
    except registry.RegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/retrain", response_model=ModelSummary)
async def retrain(
    request: RetrainRequest,
    admin_user: TokenPayload = Depends(require_admin),  # noqa: B008
) -> ModelSummary:
    await _audit(admin_user, "retrain", f"{request.sector}/{request.item}/{request.target}")
    metadata = await asyncio.to_thread(_do_retrain, request.sector, request.item, request.target)
    return _model_summary(metadata)


# --- ingest --------------------------------------------------------------


def _run_ingest(names: list[str]) -> list[IngestResultItem]:
    from ceynex.data.pipeline import run_source
    from ceynex.data.writer import UnifiedDatasetWriter, WriteResult

    writer = UnifiedDatasetWriter()
    items: list[IngestResultItem] = []
    for name in names:
        try:
            result: WriteResult = run_source(name, writer)
        except Exception as exc:  # noqa: BLE001 - one bad source must not stop the rest, see pipeline.main
            items.append(
                IngestResultItem(
                    source_id=name, status="failed", rows_in=0, rows_written=0,
                    dq_flags=0, error=str(exc),
                )
            )
            continue
        items.append(
            IngestResultItem(
                source_id=result.source_id,
                status=result.status,
                rows_in=result.rows_in,
                rows_written=result.rows_written,
                dq_flags=result.dq_flags,
                error=result.error,
                warnings=result.warnings,
            )
        )
    return items


@router.post("/pipeline/ingest", response_model=IngestResponse)
async def trigger_ingest(
    request: IngestRequest,
    admin_user: TokenPayload = Depends(require_admin),  # noqa: B008
) -> IngestResponse:
    from ceynex.data.pipeline import CONNECTORS

    names = list(CONNECTORS) if not request.sources else request.sources
    unknown = [n for n in names if n not in CONNECTORS]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unknown sources: {unknown} (known: {sorted(CONNECTORS)})",
        )

    await _audit(admin_user, "pipeline_ingest", ",".join(names))
    results = await asyncio.to_thread(_run_ingest, names)
    return IngestResponse(results=results)


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def pipeline_status(_admin: TokenPayload = Depends(require_admin)) -> PipelineStatusResponse:  # noqa: B008
    try:
        runs = await asyncio.to_thread(admin.pipeline_status)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="pipeline status unavailable") from exc
    return PipelineStatusResponse(
        runs=[
            PipelineRunItem(
                run_id=r.run_id, source_id=r.source_id, started_at=r.started_at,
                finished_at=r.finished_at, status=r.status, rows_written=r.rows_written,
                error=r.error,
            )
            for r in runs
        ]
    )


# --- DQ review -------------------------------------------------------------


@router.get("/dq-flags", response_model=DQFlagsResponse)
async def list_dq_flags(
    resolved: bool | None = None,
    severity: str | None = None,
    _admin: TokenPayload = Depends(require_admin),  # noqa: B008
) -> DQFlagsResponse:
    try:
        flags = await asyncio.to_thread(admin.list_dq_flags, resolved=resolved, severity=severity)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="dq flags unavailable") from exc
    return DQFlagsResponse(
        flags=[
            DQFlagItem(
                flag_id=f.flag_id, item=f.item, hs_code=f.hs_code, partner_iso3=f.partner_iso3,
                period_start=f.period_start, metric=f.metric, source_a=f.source_a,
                value_a=f.value_a, source_b=f.source_b, value_b=f.value_b,
                pct_diff=f.pct_diff, severity=f.severity, detected_at=f.detected_at,
                resolved=f.resolved,
            )
            for f in flags
        ]
    )


@router.post("/dq-flags/{flag_id}/resolve", response_model=ResolveDQFlagResponse)
async def resolve_dq_flag(
    flag_id: int,
    admin_user: TokenPayload = Depends(require_admin),  # noqa: B008
) -> ResolveDQFlagResponse:
    await _audit(admin_user, "resolve_dq_flag", str(flag_id))
    try:
        found = await asyncio.to_thread(admin.resolve_dq_flag, flag_id)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not resolve dq flag") from exc
    if not found:
        raise HTTPException(status_code=404, detail=f"no dq_flag with id {flag_id}")
    return ResolveDQFlagResponse(flag_id=flag_id, resolved=True)


# --- audit log (SRS 3.4.7) -------------------------------------------------


@router.get("/audit-log", response_model=AuditLogResponse)
async def audit_log(_admin: TokenPayload = Depends(require_admin)) -> AuditLogResponse:  # noqa: B008
    try:
        entries = await asyncio.to_thread(audit.list_entries)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="audit log unavailable") from exc
    return AuditLogResponse(
        entries=[
            AuditLogItem(
                id=e.id, actor_email=e.actor_email, action=e.action,
                target=e.target, logged_at=e.logged_at,
            )
            for e in entries
        ]
    )


# --- user accounts + roles (SRS 3.5.4, RBAC) -----------------------------
#
# Self-service signup (`routes/auth.py`) only ever creates an account at the
# default role. Everything that grants privilege — provisioning an account at a
# chosen role, moving an existing account between roles, disabling one — is
# here, behind `require_admin`, and every mutation writes an audit row first
# via `_audit` (SRS 3.4.7), the same as retrain/ingest/resolve above.


def _user_item(u: object) -> UserAdminItem:  # users.UserSummary | users.User
    return UserAdminItem(
        id=u.id, email=u.email, role=u.role, created_at=u.created_at, disabled=u.disabled
    )


@router.get("/users", response_model=UsersResponse)
async def list_users(_admin: TokenPayload = Depends(require_admin)) -> UsersResponse:  # noqa: B008
    try:
        entries = await asyncio.to_thread(users.list_users)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="users unavailable") from exc
    return UsersResponse(users=[_user_item(u) for u in entries])


@router.post("/users", response_model=UserMutationResponse, status_code=201)
async def create_user(
    body: CreateUserRequest,
    admin_user: TokenPayload = Depends(require_admin),  # noqa: B008
) -> UserMutationResponse:
    await _audit(admin_user, "create_user", f"{body.email} as {body.role}")
    try:
        user = await asyncio.to_thread(
            users.create_user, body.email, body.password, body.role
        )
    except users.InvalidRoleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except users.WeakPasswordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except users.EmailTakenError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not create user") from exc
    return UserMutationResponse(id=user.id, email=user.email, role=user.role, disabled=user.disabled)


@router.post("/users/{user_id}/role", response_model=UserMutationResponse)
async def set_user_role(
    user_id: int,
    body: SetRoleRequest,
    admin_user: TokenPayload = Depends(require_admin),  # noqa: B008
) -> UserMutationResponse:
    await _audit(admin_user, "set_user_role", f"user {user_id} -> {body.role}")
    try:
        user = await asyncio.to_thread(users.set_role, user_id, body.role)
    except users.InvalidRoleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except users.LastAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not change role") from exc
    if user is None:
        raise HTTPException(status_code=404, detail=f"no user with id {user_id}")
    return UserMutationResponse(id=user.id, email=user.email, role=user.role, disabled=user.disabled)


async def _set_disabled(user_id: int, admin_user: TokenPayload, *, disabled: bool) -> UserMutationResponse:
    await _audit(admin_user, "disable_user" if disabled else "enable_user", f"user {user_id}")
    try:
        user = await asyncio.to_thread(users.set_disabled, user_id, disabled=disabled)
    except users.LastAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not update user") from exc
    if user is None:
        raise HTTPException(status_code=404, detail=f"no user with id {user_id}")
    return UserMutationResponse(id=user.id, email=user.email, role=user.role, disabled=user.disabled)


@router.post("/users/{user_id}/disable", response_model=UserMutationResponse)
async def disable_user(
    user_id: int,
    admin_user: TokenPayload = Depends(require_admin),  # noqa: B008
) -> UserMutationResponse:
    return await _set_disabled(user_id, admin_user, disabled=True)


@router.post("/users/{user_id}/enable", response_model=UserMutationResponse)
async def enable_user(
    user_id: int,
    admin_user: TokenPayload = Depends(require_admin),  # noqa: B008
) -> UserMutationResponse:
    return await _set_disabled(user_id, admin_user, disabled=False)
