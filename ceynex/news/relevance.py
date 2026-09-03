"""Supports docs/ARCHITECTURE_DELTA.md D11 — ranking headlines against a question.

GDELT's `sort=HybridRel` returns an *order* and no score. An order is not enough:
the panel needs a floor (some questions have no relevant coverage, and the least
irrelevant headline is not an answer) and the UI needs something to label a row
with. So the ordering is redone locally with the cross-encoder that
`retrieval/client.py` already holds open — `shared_models()` exists for this, and
loading a second copy of identical weights would be the only real cost here.

Why this floor is not `MIN_RERANK_SCORE`
----------------------------------------
`retrieval/client.py` sets its floor at 0.0 and argues the case carefully.
**Neither half of that argument transfers**, which is why reusing the constant
would be a mistake rather than a shortcut:

1. *The inputs are a different length.* That floor scores a question against a
   multi-sentence passage. This scores a question against a 5-12 word headline.
   `ms-marco-MiniLM` was trained on MSMARCO passages of ~50-60 tokens; a headline
   is a fifth of that and there is proportionally less lexical overlap to reward,
   so titles score systematically lower on the same model. Applying 0.0 here
   empties the panel for most real questions.

2. *A false positive costs something different.* A weakly-related policy chunk
   becomes a citation attached to a claim it does not support — invisible to
   `orchestrator/grounding.py`, and the reason that floor is strict. A weakly
   related headline is a link in a panel that says on its face it was not used to
   produce the answer. The false positive costs one mediocre link; the false
   negative costs an empty panel.

So the floor lives in `config/news.yaml`, where the repo puts numbers that are
judgement rather than derivation. **It is empirical, not calibrated** — see
`docs/DEFERRED.md`. Measured over the first live sweep of ten real questions
against the watchlist topics, responsive headlines score roughly -4 to +3 while
plainly unrelated ones sit below -8; -6.0 sits in the gap and is deliberately
generous, because this panel is allowed to be loose in a way policy evidence is
not.

Degrading
---------
Without the `[policy]` extra there is no fastembed and no cross-encoder. That is
not an error: the articles are real, GDELT already ordered them by its own
relevance, and they go back unscored with `relevance=None` so the UI omits the
badge rather than inventing one. It is a genuinely useful mode, and it is what
keeps the extra defensibly optional.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from ceynex import settings
from ceynex.news.schema import NewsArticle

log = logging.getLogger(__name__)

#: Used only when config/news.yaml is unreadable. The config is the real home.
FALLBACK_MIN_SCORE = -6.0


async def _default_provider() -> dict[str, Any]:
    from ceynex.retrieval.client import shared_models  # noqa: PLC0415 - optional extra

    return await shared_models()


_provider: Callable[[], Awaitable[dict[str, Any]]] = _default_provider


def set_models_provider(provider: Callable[[], Awaitable[dict[str, Any]]] | None) -> None:
    """Test seam, same shape as `routes/query.py`'s `set_window()`.

    Production never calls this. Passing None restores the real provider.
    """
    global _provider  # noqa: PLW0603 - one process-lifetime indirection
    _provider = provider if provider is not None else _default_provider


def min_score() -> float:
    """The floor, from config. Read per call so a restart is the only ceremony."""
    try:
        return float(settings.news_config()["relevance"]["min_score"])
    except (KeyError, TypeError, ValueError, FileNotFoundError):
        log.warning("could not read relevance.min_score from config/news.yaml; using the fallback")
        return FALLBACK_MIN_SCORE


async def score_articles(
    query: str,
    articles: Sequence[NewsArticle],
    *,
    limit: int,
    floor: float | None = None,
) -> list[NewsArticle]:
    """Rank `articles` against `query`, drop the ones below the floor, keep `limit`.

    Returns articles in GDELT's own order, unscored, when no cross-encoder is
    available. Never raises: a scoring failure must cost the ranking, not the
    panel.
    """
    if not articles:
        return []

    try:
        models = await _provider()
    except ImportError:
        log.info("fastembed not installed — news results are unranked (pip install '.[policy]')")
        return list(articles)[:limit]
    except Exception:  # noqa: BLE001 - degrading to GDELT's own order is the contract
        log.warning("could not load the cross-encoder; news results are unranked", exc_info=True)
        return list(articles)[:limit]

    try:
        # fastembed is synchronous ONNX, same as `retrieval/client.py`'s `_rerank`.
        scored = await asyncio.to_thread(_rerank_titles, models, query, list(articles))
    except Exception:  # noqa: BLE001 - a scoring failure must not cost the panel
        log.warning("headline scoring failed; news results are unranked", exc_info=True)
        return list(articles)[:limit]

    cut = min_score() if floor is None else floor
    kept = [article for article in scored if (article.relevance or 0.0) >= cut][:limit]
    if scored and not kept:
        log.debug(
            "every headline scored below %.1f (best %.2f) — returning nothing",
            cut,
            scored[0].relevance or 0.0,
        )
    return kept


def _rerank_titles(
    models: dict[str, Any], query: str, articles: list[NewsArticle]
) -> list[NewsArticle]:
    """Cross-encoder over (question, headline) pairs, highest first.

    Scores are replaced rather than blended with anything: there is nothing to
    blend, since GDELT gave an order and no number.
    """
    scores = list(models["rerank"].rerank(query, [article.embed_text() for article in articles]))
    ranked = [
        NewsArticle(**{**vars(article), "relevance": float(score)})
        for article, score in zip(articles, scores, strict=False)
    ]
    return sorted(ranked, key=lambda article: article.relevance or 0.0, reverse=True)


__all__ = ["FALLBACK_MIN_SCORE", "min_score", "score_articles", "set_models_provider"]
