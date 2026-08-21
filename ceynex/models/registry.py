"""Implements SRS 3.1.10 — model versioning, storage and retraining (SAD Figure 10).

Three members train models into one registry: M1's agriculture models, M3's
apparel models, and M2's baselines. The `forecast` agent serves whatever is
registered for an item without knowing who trained it or how, which is why the
lookup is by `{sector}/{item}/{target}` rather than by owner.

Layout on disk::

    models/
      agriculture/cinnamon/export_value_usd/
        v20260819T101500Z/
          model.pkl
          metadata.json
        latest -> v20260819T101500Z

Versions are UTC timestamps so they sort chronologically, but `latest` is
resolved from each version's recorded `saved_at` rather than from the directory
name — a caller is free to pass `version="baseline"` and the registry still
knows which one is newest.

**Artifacts are pickles.** That is a deliberate, bounded choice: the alternative
is a per-model-family serializer for statsmodels, LightGBM and whatever M1 and
M3 subclass next, which is a lot of code to load files this project writes
itself. It also means a model file is executable content — never load a registry
directory that came from outside the team.
"""

from __future__ import annotations

import contextlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ceynex.contracts import ForecastModel
from ceynex.settings import models_dir

log = logging.getLogger(__name__)

ARTIFACT = "model.pkl"
METADATA = "metadata.json"
LATEST = "latest"

# The band `ForecastPoint.lower/upper` describe unless a model says otherwise.
DEFAULT_INTERVAL_LEVEL = 0.80


class RegistryError(RuntimeError):
    """Raised when a requested model is not in the registry."""


@dataclass(frozen=True)
class ModelMetadata:
    """What `metadata.json` records about one trained version.

    `metrics` comes from the shared rolling-origin harness in `eval/backtest.py`,
    so MAPE from M1's model and MAPE from M3's model mean the same thing. A
    version saved without metrics is legal — it is how a model looks between
    being fitted and being backtested — and `list_models` shows the gap rather
    than hiding it.
    """

    sector: str
    item: str
    target: str
    version: str
    saved_at: str  # ISO-8601 UTC
    model_class: str
    model_module: str
    training_rows: int | None = None
    params: dict[str, Any] | None = None
    metrics: dict[str, float] | None = None
    interval_level: float = DEFAULT_INTERVAL_LEVEL
    notes: str | None = None

    @property
    def model_id(self) -> str:
        """The string agents put in `Evidence.detail`. Includes the version.

        Without the version an evidence trail says a model produced a number but
        not *which* model, which is exactly the traceability the evidence is for.
        """
        return f"{self.sector}/{self.item}/{self.target}@{self.version}"


# --- paths ---------------------------------------------------------------


def _root() -> Path:
    """Registry root. From settings, never a path relative to this file.

    Three separate deployment failures in this project traced to modules
    resolving data through `__file__`; `tests/test_layout.py` now fails on the
    pattern.
    """
    return models_dir()


def _model_dir(sector: str, item: str, target: str) -> Path:
    return _root() / _slug(sector) / _slug(item) / _slug(target)


def _slug(value: str) -> str:
    """Path-safe, lowercase. `Apparel Knit` and `apparel_knit` are one model."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value).strip().lower())


def _new_version() -> str:
    return datetime.now(UTC).strftime("v%Y%m%dT%H%M%SZ")


def _free_version(parent: Path) -> str:
    """A timestamp version that is not already taken.

    Timestamps are second-granular, and two models registered in the same second
    is not hypothetical — it is what a loop over model families does, and the
    second save would otherwise overwrite the first with no error and no trace.
    """
    base = _new_version()
    if not (parent / base).exists():
        return base
    for suffix in range(2, 100):
        candidate = f"{base}-{suffix}"
        if not (parent / candidate).exists():
            return candidate
    raise RegistryError(f"could not find a free version under {parent}")


# --- writing -------------------------------------------------------------


def save(
    model: ForecastModel,
    *,
    version: str | None = None,
    training_rows: int | None = None,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    interval_level: float = DEFAULT_INTERVAL_LEVEL,
    notes: str | None = None,
) -> ModelMetadata:
    """Persist a fitted model and return its metadata.

    `sector`, `item` and `target` come off the model itself — they are declared
    on the `ForecastModel` contract precisely so the registry does not need them
    passed separately and cannot be told a different answer than the model holds.
    """
    for attribute in ("sector", "item", "target"):
        if not getattr(model, attribute, None):
            raise RegistryError(
                f"model does not declare `{attribute}`; the registry keys on it "
                "and cannot file a model that has not said what it forecasts"
            )

    parent = _model_dir(model.sector, model.item, model.target)
    if version is None:
        version = _free_version(parent)
    directory = parent / version
    directory.mkdir(parents=True, exist_ok=True)

    metadata = ModelMetadata(
        sector=model.sector,
        item=model.item,
        target=model.target,
        version=version,
        saved_at=datetime.now(UTC).isoformat(),
        model_class=type(model).__name__,
        model_module=type(model).__module__,
        training_rows=training_rows,
        params=params if params is not None else _params_of(model),
        metrics=metrics,
        interval_level=interval_level,
        notes=notes,
    )

    # Artifact first: a metadata.json with no model beside it would make the
    # registry advertise something it cannot serve.
    (directory / ARTIFACT).write_bytes(pickle.dumps(model))
    _write_metadata(directory, metadata)
    _point_latest_at(directory.parent, version)

    log.info("registered %s", metadata.model_id)
    return metadata


def record_metrics(metadata: ModelMetadata, metrics: dict[str, float]) -> ModelMetadata:
    """Attach backtest metrics to an already-saved version.

    Split from `save` because fitting and backtesting are separate steps: the
    harness needs a saved model to evaluate, so the metrics cannot exist at the
    moment the model is written.
    """
    updated = ModelMetadata(**{**asdict(metadata), "metrics": dict(metrics)})
    directory = _model_dir(metadata.sector, metadata.item, metadata.target) / metadata.version
    if not directory.exists():
        raise RegistryError(f"{metadata.model_id} is not in the registry")
    _write_metadata(directory, updated)
    return updated


def _write_metadata(directory: Path, metadata: ModelMetadata) -> None:
    (directory / METADATA).write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True))


def _point_latest_at(parent: Path, version: str) -> None:
    """Best-effort `latest` symlink.

    Convenience for a human reading the directory, not the mechanism `load`
    relies on — symlink creation fails on Windows without developer mode, and a
    teammate on Windows should still get a working registry.
    """
    link = parent / LATEST
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(version, target_is_directory=True)
    except OSError as exc:
        log.debug("could not create `latest` symlink in %s: %s", parent, exc)


def _params_of(model: ForecastModel) -> dict[str, Any] | None:
    """Ask the model to describe its own hyperparameters, if it can."""
    describe = getattr(model, "describe_params", None)
    if callable(describe):
        try:
            return dict(describe())
        except Exception as exc:  # noqa: BLE001 - metadata is not worth failing a save over
            log.debug("describe_params failed for %s: %s", type(model).__name__, exc)
    return None


# --- reading -------------------------------------------------------------


def load(sector: str, item: str, target: str, version: str = LATEST) -> ForecastModel:
    """Load one model. `version="latest"` resolves to the most recently saved."""
    parent = _model_dir(sector, item, target)
    if version == LATEST:
        resolved = _latest_version(parent)
        if resolved is None:
            raise RegistryError(f"no versions registered under {sector}/{item}/{target}")
        version = resolved

    artifact = parent / version / ARTIFACT
    if not artifact.exists():
        raise RegistryError(f"{sector}/{item}/{target}@{version} is not in the registry")

    model = pickle.loads(artifact.read_bytes())
    # Stamp the version onto the instance so an agent can name it in evidence
    # without a second registry call.
    with contextlib.suppress(AttributeError):  # a model using __slots__
        model.version = version
    return model


def load_latest(
    *, item: str, sector: str | None = None, target: str | None = None
) -> ForecastModel | None:
    """The newest model for an item, or None if nobody has registered one.

    None rather than an exception because "no model yet" is the normal state for
    most of this sprint: the forecast agent falls back to its drift baseline and
    keeps answering. A missing model is a reason to be humble, not to fail.
    """
    candidates = [
        m
        for m in list_models()
        if _slug(m.item) == _slug(item)
        and (sector is None or _slug(m.sector) == _slug(sector))
        and (target is None or _slug(m.target) == _slug(target))
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda m: m.saved_at)
    return load(newest.sector, newest.item, newest.target, newest.version)


def load_best(
    *,
    item: str,
    sector: str | None = None,
    target: str | None = None,
    metric: str = "mape",
) -> ForecastModel | None:
    """The best-scoring registered model for an item, or None if none is scored.

    `load_latest` answers "what was registered most recently", which is the wrong
    question when two families are registered for the same item: measured on real
    Sri Lankan data, SARIMA scores 6.3% MAPE on cinnamon and LightGBM 12.2%, and
    serving the worse one because it happened to be saved second is a silent
    accuracy loss with no signal that it happened.

    Only versions carrying a backtest metric are eligible — an unscored model has
    made no claim to be better, so it does not get to win by default.
    """
    scored = [
        m
        for m in list_models()
        if _slug(m.item) == _slug(item)
        and (sector is None or _slug(m.sector) == _slug(sector))
        and (target is None or _slug(m.target) == _slug(target))
        and m.metrics is not None
        and _usable(m.metrics.get(metric))
    ]
    if not scored:
        return None
    best = min(scored, key=lambda m: m.metrics[metric])
    return load(best.sector, best.item, best.target, best.version)


def _usable(value: Any) -> bool:
    """A metric that is missing or NaN cannot rank anything."""
    return isinstance(value, int | float) and value == value


def list_models(sector: str | None = None, item: str | None = None) -> list[ModelMetadata]:
    """Every registered version, newest first. Empty registry is not an error."""
    root = _root()
    if not root.exists():
        return []

    found: list[ModelMetadata] = []
    for path in sorted(root.glob(f"*/*/*/*/{METADATA}")):
        # `latest` is a symlink into a sibling version directory, so globbing
        # finds every registered version twice — once by name and once through
        # the link. Counting a model twice would skew any "how many models are
        # registered" claim in the evaluation report.
        if path.parent.is_symlink():
            continue
        metadata = _read_metadata(path)
        if metadata is None:
            continue
        if sector and _slug(metadata.sector) != _slug(sector):
            continue
        if item and _slug(metadata.item) != _slug(item):
            continue
        found.append(metadata)
    return sorted(found, key=lambda m: m.saved_at, reverse=True)


def _read_metadata(path: Path) -> ModelMetadata | None:
    try:
        raw = json.loads(path.read_text())
        return ModelMetadata(**{k: v for k, v in raw.items() if k in ModelMetadata.__annotations__})
    except (OSError, ValueError, TypeError) as exc:
        # One unreadable version must not hide every other model in the registry.
        log.warning("skipping unreadable model metadata at %s: %s", path, exc)
        return None


def _latest_version(parent: Path) -> str | None:
    versions = [
        _read_metadata(d / METADATA)
        for d in parent.iterdir()
        if d.is_dir() and not d.is_symlink() and (d / METADATA).exists()
    ] if parent.exists() else []
    real = [v for v in versions if v is not None]
    if not real:
        return None
    return max(real, key=lambda m: m.saved_at).version


# --- retraining ----------------------------------------------------------


def retrain(
    sector: str,
    item: str,
    target: str,
    df: pd.DataFrame,
    *,
    notes: str | None = None,
) -> ModelMetadata:
    """Refit the registered model on new data and save it as a new version.

    The hook behind M1's admin retrain endpoint (SRS 3.5.4). It re-fits the same
    model *class* with the same hyperparameters rather than re-selecting a model,
    so a retrain is a data refresh and not a silent architecture change — the
    previous version stays on disk and stays loadable if the new one is worse.
    """
    previous = load(sector, item, target)
    # describe_params() already carries sector/item/target, so they are merged
    # rather than passed alongside — passing both is a duplicate-keyword TypeError.
    params = {**(_params_of(previous) or {}), "sector": sector, "item": item, "target": target}
    fitted = type(previous)(**params)
    fitted.fit(df)
    return save(
        fitted,
        training_rows=len(df),
        notes=notes or f"retrained from {sector}/{item}/{target}",
    )


__all__ = [
    "DEFAULT_INTERVAL_LEVEL",
    "ModelMetadata",
    "RegistryError",
    "list_models",
    "load",
    "load_best",
    "load_latest",
    "record_metrics",
    "retrain",
    "save",
]
