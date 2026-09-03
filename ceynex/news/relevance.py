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
judgement rather than derivation.

The measurement
---------------
Eleven questions against the live `ceynex_news` collection on the deployed box,
scoring **the candidates the vector search returns** — the same path a request
takes, `store.search()` then this function:

    sri lanka coconut exports          +7.16      who won the cricket    -9.94
    sri lanka apparel exports          +0.65      rain in Colombo        -8.96
    how are apparel exports doing?     -1.64      chocolate cake        -10.93
    what tariffs do exports face?      -3.33
    shipping costs for exporters?      -6.66
    how is Ceylon tea performing?      -7.76
    global commodity prices            -8.92
    ceylon tea auction prices         -10.72

`min_score: -8.0` is the loosest cut that still rejects every out-of-scope
question: it keeps 6 of 8 in-scope and admits 0 of 3 out. -9.0 would keep 7 but
let "will it rain in Colombo tomorrow" into a trade panel.

**Measure the path that runs, not the one that is easy to measure.** The first
attempt scored every indexed title against each question and concluded that all
scores are negative, best -1.78. That is true of the corpus and false of the
route, which never sees a title the vector search did not retrieve. A floor set
from it emptied the panel for every question in production. The difference is
not subtle — +7.16 against -1.78 — and it was only visible from the deployed
box, because it depends on what is in the collection.

**The ranges overlap, and no floor fixes that.** "ceylon tea auction prices"
scores -10.72 — below every out-of-scope question — because this 150-headline
corpus has no tea-auction story in it. That is a corpus problem, not a threshold
problem. Re-measure once the refresher has run for a few days.
Recorded in `docs/DEFERRED.md`.

The same run set the label boundaries in `schema.py`, which had the same bug in
reverse: they cut on `sigmoid(logit)` at 0.5 and 0.1 — logits 0.0 and -2.2 —
picked from the model's training distribution rather than from anything
observed here.

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
