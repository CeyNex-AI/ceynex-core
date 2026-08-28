"""Runtime configuration. Environment for credentials, YAML for anything tuneable.

The split is deliberate. Credentials differ per machine and must never be
committed, so they come from the environment. Model ids, elasticity assumptions
and timeouts are decisions a marker may ask us to justify, so they live in
`config/*.yaml` under version control where a diff shows who changed what.

Two deployment shapes have to work from the same code:

- a developer's `make up` stack, which sets POSTGRES_HOST/PORT/USER/... separately
- the backend VM, whose compose hands over a single POSTGRES_URL

`postgres_dsn()` accepts either.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    """Where `config/*.yaml` lives, in a source checkout or an installed package.

    Not a constant, because the answer differs between the two layouts and the
    difference only shows up on deployment. In a checkout the files sit beside
    the `ceynex` package; installed into site-packages, `__file__.parent.parent`
    is site-packages itself and the config is wherever the image put it — /app
    for our container. Candidates are tried in order of specificity.
    """
    override = os.environ.get("CEYNEX_CONFIG_DIR")
    candidates = [
        Path(override) if override else None,
        Path.cwd() / "config",
        REPO_ROOT / "config",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "llm.yaml").is_file():
            return candidate

    searched = [str(c) for c in candidates if c is not None]
    raise FileNotFoundError(
        "could not find config/llm.yaml. Looked in: "
        + ", ".join(searched)
        + ". Set CEYNEX_CONFIG_DIR to point at it."
    )


# Loaded once, and never overriding what is already exported: a value set in the
# real environment (the container, CI) must win over a stray local .env.
load_dotenv(REPO_ROOT / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else value


def postgres_dsn() -> str:
    """A libpq connection string, from POSTGRES_URL or assembled from the parts."""
    url = _env("POSTGRES_URL")
    if url:
        return url
    user = _env("POSTGRES_USER", "ceynex")
    password = _env("POSTGRES_PASSWORD", "ceynex_dev")
    host = _env("POSTGRES_HOST", "localhost")
    port = _env("POSTGRES_PORT", "5432")
    database = _env("POSTGRES_DB", "ceynex")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def redacted_dsn(dsn: str | None = None) -> str:
    """The DSN with its password replaced, for logs and error messages."""
    dsn = dsn or postgres_dsn()
    if "://" not in dsn or "@" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    credentials, location = rest.rsplit("@", 1)
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:***@{location}"


def neo4j_config() -> tuple[str, str, str]:
    """(uri, user, password)."""
    return (
        _env("NEO4J_URI", "bolt://localhost:7687") or "bolt://localhost:7687",
        _env("NEO4J_USER", "neo4j") or "neo4j",
        _env("NEO4J_PASSWORD", "ceynex_dev_pw") or "ceynex_dev_pw",
    )


def openai_api_key() -> str | None:
    """None is a supported state — the system degrades rather than failing (SRS 3.4.3)."""
    return _env("OPENAI_API_KEY") or None


def openrouter_api_key() -> str | None:
    """R5 failsafe provider key. None just means the failsafe is unusable, same
    degrade-don't-fail posture as `openai_api_key()`."""
    return _env("OPENROUTER_API_KEY") or None


def jwt_secret() -> str:
    """Signing key for auth tokens (SRS 3.1.11).

    The fallback keeps `make up`/tests working without a `.env` entry, same
    spirit as `postgres_dsn()`'s dev defaults. Unlike a DB password, this one
    is not optional in a real deployment: anyone who knows it can forge a
    token for any of the four demo accounts. Set CEYNEX_JWT_SECRET wherever
    the API is reachable outside a developer's own machine.
    """
    return _env("CEYNEX_JWT_SECRET") or "ceynex-dev-insecure-jwt-secret-change-me"


def comtrade_api_key() -> str | None:
    """None falls the connector back to its committed bulk extract (R2)."""
    return _env("COMTRADE_API_KEY") or None


def redis_url() -> str | None:
    """Shared store for the SRS 3.4.6 rate limiter.

    The backend VM's compose has always passed this; nothing read it until the
    limiter existed. None is supported — the limiter falls back to a
    per-process counter, which is correct for a single-worker developer run and
    understated for the deployed two-worker image (see
    `ceynex/api/rate_limit.py`).
    """
    return _env("REDIS_URL") or None


def data_dir() -> Path:
    return Path(_env("CEYNEX_DATA_DIR", str(REPO_ROOT / "data")) or "data")


def models_dir() -> Path:
    return Path(_env("CEYNEX_MODELS_DIR", str(REPO_ROOT / "models")) or "models")


@functools.lru_cache(maxsize=8)
def load_config(name: str) -> dict[str, Any]:
    """Read `config/<name>.yaml`. Cached — these are read on every agent call."""
    path = config_dir() / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def llm_config() -> dict[str, Any]:
    """SRS 3.6.5 — provider and model ids, kept out of code so cost can be traded
    against quality without a rebuild."""
    return load_config("llm")


def elasticity_config() -> dict[str, Any]:
    """SRS 3.1.5 — the simulation assumptions, in a table a marker can read."""
    return load_config("elasticities")
