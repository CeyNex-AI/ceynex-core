"""Site-wide settings -- the optional "Signal Deck" UI theme (2026-09-04).

`GET /api/site/theme` is public and unauthenticated on purpose: every visitor's
page load needs to know which theme to render, signed in or not, including on
the sign-in page itself. `POST /api/site/theme` requires the admin role -- this
changes what every other visitor sees, not just the caller's own view, so it
gets the same `require_admin` gate as the rest of SRS 3.5.4's admin surface,
and the same audit rule (SRS 3.4.7): the row is written before the change, and
a failed write refuses the change rather than letting it run unlogged.
"""

from __future__ import annotations

import asyncio

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from ceynex.api import audit, site_settings
from ceynex.api.routes.auth import TokenPayload, require_admin
from ceynex.api.schemas import SetSiteThemeRequest, SiteThemeResponse

router = APIRouter(prefix="/api/site", tags=["site"])


@router.get("/theme", response_model=SiteThemeResponse)
async def get_theme() -> SiteThemeResponse:
    theme = await asyncio.to_thread(site_settings.get_theme)
    return SiteThemeResponse(theme=theme)


@router.post("/theme", response_model=SiteThemeResponse)
async def set_theme(
    request: SetSiteThemeRequest,
    admin: TokenPayload = Depends(require_admin),  # noqa: B008
) -> SiteThemeResponse:
    if request.theme not in site_settings.VALID_THEMES:
        # Checked before the audit row, so a typo is not logged as a change.
        raise HTTPException(
            status_code=422,
            detail=f"unknown theme {request.theme!r}; must be one of {site_settings.VALID_THEMES}",
        )
    try:
        await asyncio.to_thread(
            audit.record, actor_email=admin.email, action="set_site_theme", target=request.theme
        )
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=503, detail="could not write audit log; action not performed"
        ) from exc
    try:
        theme = await asyncio.to_thread(
            site_settings.set_theme, request.theme, updated_by=admin.email
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not save site theme") from exc
    return SiteThemeResponse(theme=theme)
