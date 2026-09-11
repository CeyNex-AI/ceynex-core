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


def qdrant_url() -> str | None:
    """Where the policy-document vector store lives (SRS 3.1.9, deviation D10).

    None is a supported state, same posture as `redis_url()` and
    `openai_api_key()`: with no Qdrant configured the Trade Economics agent
    simply does not retrieve policy text and answers exactly as it did before
    retrieval existed. A missing datastore must never be the thing that fails a
    query.
    """
    url = _env("QDRANT_URL")
    if url:
        return url
    host = _env("QDRANT_HOST")
    port = _env("QDRANT_PORT")
    if host or port:
        return f"http://{host or 'localhost'}:{port or '6333'}"
    return None


def qdrant_collection() -> str:
    """The collection the offline pipeline indexes into.

    Deliberately not `ceynex`, which is the prototype's 384-dimension
    page-per-vector collection. A collection's vector size cannot be changed in
    place, so the new embedding model needs a new collection rather than a
    migration.
    """
    return _env("QDRANT_COLLECTION", "ceynex_policy") or "ceynex_policy"


def news_collection() -> str:
    """Where the news sidecar's headlines live (docs/ARCHITECTURE_DELTA.md D11).

    Deliberately a *second* collection rather than a `doc_type` field on
    `ceynex_policy`. The policy corpus is hand-curated and human-verified because
    anything cited as evidence has to be checkable (SRS 3.1.4); news is
    high-volume and unvetted. Sharing a collection would mean every existing
    retrieval path had to remember to exclude news, and forgetting once is a
    silent failure — the exact shape `retrieval/schema.py` was written to
    prevent. Same embedding model and dimensions, so the loaded ONNX sessions
    are shared and the second collection costs nothing at query time.
    """
    return _env("QDRANT_NEWS_COLLECTION", "ceynex_news") or "ceynex_news"


def news_enabled() -> bool:
    """Kill switch for the whole news sidecar.

    Same rule as `policy_retrieval_enabled()`: any value other than
    "off"/"0"/"false"/"no" leaves it on, because a typo in an env var must not
    silently disable a feature.
    """
    value = (_env("CEYNEX_NEWS", "on") or "on").lower()
    return value not in ("off", "0", "false", "no")


def news_refresh_enabled() -> bool:
    """Kill switch for the hourly refresher alone, separate from serving news.

    Two switches rather than one so a second deployment can serve the trending
    panel from the shared snapshot without also fetching from GDELT — the
    refresher is the only part with an outbound call budget to protect.
    """
    value = (_env("CEYNEX_NEWS_REFRESH", "on") or "on").lower()
    return value not in ("off", "0", "false", "no")


def news_cache_dir() -> Path:
    """Where GDELT responses are cached. Inside the existing `llm_cache` volume."""
    return Path(_env("CEYNEX_NEWS_CACHE_DIR", str(REPO_ROOT / ".cache" / "news")) or ".cache/news")


def news_config() -> dict[str, Any]:
    """D11 — the watchlist and the tuning knobs, in a table a marker can read."""
    return load_config("news")


def news_base_url() -> str:
    """The GDELT DOC 2.0 endpoint, env-overridable per deployment.

    Normally a URL like this belongs in YAML and only in YAML — it is a tunable,
    not a credential. This one gets an environment override because it has to
    differ *per machine*: `api.gdeltproject.org` refuses TLS from some networks
    (the deployed backend VM among them) while answering happily on port 80, and
    `config/` ships baked into the image, so a committed value cannot vary by
    host. Same reasoning as `postgres_dsn()` accepting either a URL or parts.
    """
    override = _env("CEYNEX_GDELT_BASE_URL")
    if override:
        return override
    try:
        return str(news_config()["gdelt"]["base_url"])
    except (KeyError, TypeError, FileNotFoundError):
        return "https://api.gdeltproject.org/api/v2/doc/doc"


def policy_retrieval_enabled() -> bool:
    """Kill switch for the retrieval path, so the evaluation can measure without it.

    `make eval-policy-baseline` sets CEYNEX_POLICY_RETRIEVAL=off to produce the
    before-figures that the after-figures are only meaningful against. Any value
    other than "off"/"0"/"false" leaves it on, because a typo in an env var must
    not silently disable a feature.
    """
    value = (_env("CEYNEX_POLICY_RETRIEVAL", "on") or "on").lower()
    return value not in ("off", "0", "false", "no")


def chat_enabled() -> bool:
    """Kill switch for the conversational surface.

    Off leaves `POST /api/query` answering exactly as it always has — the same
    posture `policy_retrieval_enabled` takes, and for the same reason: a new
    surface has to be removable without touching the one the evaluation measures.
    """
    value = (_env("CEYNEX_CHAT", "on") or "on").lower()
    return value not in ("off", "0", "false", "no")


def clarify_enabled() -> bool:
    """Whether the clarification gate may ask before answering.

    Separate from `chat_enabled` on purpose: a demo may want conversation without
    ever being interrupted by a question, and a reviewer comparing answers
    against `queries.md` needs to turn it off without losing chat.
    """
    value = (_env("CEYNEX_CLARIFY", "on") or "on").lower()
    return value not in ("off", "0", "false", "no")


def scenario_enabled() -> bool:
    """Kill switch for the scenario workbench (deviation D17).

    Off removes `/api/scenario/*` and nothing else: the trade-economics agent
    keeps simulating from questions exactly as before, because the formulas it
    shares with the workbench live in `models/shocks.py` either way.
    """
    value = (_env("CEYNEX_SCENARIO", "on") or "on").lower()
    return value not in ("off", "0", "false", "no")


def web_search_enabled() -> bool:
    """Whether general web search may run at all (deviation D14).

    Distinct from having a key: `tavily_api_key()` being None falls back to the
    keyless provider, whereas this being off means no outbound search happens on
    any provider. `off` is what makes "answers are byte-identical to the
    pre-web-search system" a checkable claim.
    """
    value = (_env("CEYNEX_WEB_SEARCH", "on") or "on").lower()
    return value not in ("off", "0", "false", "no")


def tavily_api_key() -> str | None:
    """None is a supported state, not an error — the keyless tier covers it."""
    return _env("TAVILY_API_KEY") or None


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


def citations_enabled() -> bool:
    """Inline `[n]` citation markers in merge prose (execution plan §7).

    **Default off**, unlike the other feature switches here, and deliberately so.
    Turning it on changes the prompt every answer is written from, and
    `docs/EVALUATION.md` §8 is explicit that an unmeasured prompt change is worth
    nothing until it has been run against the 30-question set. The flag exists so
    that run is a comparison rather than a leap.
    """
    value = (_env("CEYNEX_CITATIONS", "off") or "off").lower()
    return value in ("on", "1", "true", "yes")
