"""Baseline comparison: does grounding answers in the knowledge graph and agents
reduce unsupported figures, compared with asking a language model directly?

    python -m eval.baseline --repeat 3 --out eval/results/baseline
    python -m eval.baseline --id S01          # one question, every condition

Runs every question in `eval/questions.yaml` under four conditions:

- **A: LLM only.** The merge model (gpt-4o) answers the question directly, with no
  data and no tools. It is told to say so when it does not know, so it is not
  set up to fail.
- **B: CeyNex, no guard.** The full pipeline, but both grounding guards (the
  merge guard in `orchestrator.merger` and each agent's explanation guard in
  `agents.common`) are disabled in-process. Run straight after C against the
  same fresh prompt cache, so its model calls replay C's: the only difference
  between B and C is the guard.
- **C: CeyNex.** The full system, exactly as `eval.harness` runs it.
- **D: CeyNex, deterministic.** No language model at all (SRS 3.4.3's degraded
  path, keyword router and composer).

**Two levels.** *Dataset*: the figure appears in the evidence or in a figure
an agent computed from the data (the guard's own corpus), which is the fair
comparison with a model that has no data. *Strict*: the figure appears in a
cited evidence record, i.e. a reader can find it in the evidence panel; derived
values such as a difference between two cited prices fail this level.

**How a figure is judged.** A plain model has no evidence of its own, so every
condition is scored against one shared reference: the question plus every
evidence entry that the CeyNex conditions (B, C and D) retrieved for that
question in the same repeat, i.e. everything the project's dataset returned. A
figure absent from it is *unsupported by the dataset*. That is not the same as
*wrong*: a plain model may quote a real figure from outside the dataset. The
metric measures traceability, which is the system's claim, and the per-answer
records are written out so a sample can be checked by hand.

The comparison uses `grounding.ungrounded_figures`, the same rule the runtime
guards use, so bare calendar years and one- or two-digit structural numbers
are never counted as figures.

**Isolation from production.** When run inside the deployed API container, the
spend counter is replaced by an in-process one and the caps are lifted, so the
run neither consumes nor is throttled by the deployment's daily allowance. The
prompt cache is redirected to a private directory that is cleared before each
repeat.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from ceynex.orchestrator import grounding
from eval.harness import GuardRecords, is_refusal, load_questions

log = logging.getLogger(__name__)

CONDITIONS = ("C", "B", "D", "A")
LABELS = {
    "A": "LLM only (gpt-4o, no data)",
    "B": "CeyNex without grounding guard",
    "C": "CeyNex (full)",
    "D": "CeyNex deterministic (no LLM)",
}
A_MODEL = "gpt-4o"
A_TEMPERATURE = 0.2  # the merge role's temperature in config/llm.yaml
A_SYSTEM = (
    "You are a trade analyst specialising in Sri Lanka's exports of tea, cinnamon, rubber, "
    "coconut and apparel. Answer the user's question accurately and concisely, in at most "
    "150 words, giving specific figures where you can. If you do not know, or the data is "
    "not available, say so plainly rather than guessing."
)
CACHE_DIR = Path(tempfile.gettempdir()) / "ceynex_eval_baseline_cache"


# --------------------------------------------------------------------- isolation
def isolate() -> None:
    """Private prompt cache, private spend counter, no caps."""
    from ceynex.llm import client as llm_client
    from ceynex.observability import spend

    original_init = llm_client.PromptCache.__init__

    def private_cache(self, path, ttl_hours=168.0, enabled=True):  # noqa: ANN001
        original_init(self, CACHE_DIR, ttl_hours=ttl_hours, enabled=enabled)

    llm_client.PromptCache.__init__ = private_cache
    spend.set_shared_counter(spend.InProcessSpendCounter())

    original_limits = llm_client.LLMReasoningClient._limits

    def uncapped(self):  # noqa: ANN001
        limits = dict(original_limits(self))
        limits["daily_spend_cap_usd"] = 0
        limits["per_user_daily_cap_usd"] = 0
        return limits

    llm_client.LLMReasoningClient._limits = uncapped


def clear_cache() -> None:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


class GuardsOff:
    """Condition B: both grounding guards report nothing while active."""

    def __enter__(self) -> GuardsOff:
        from ceynex.agents import common
        from ceynex.orchestrator import merger

        self._saved = (merger._reject_ungrounded_prose, common.ungrounded_figures)
        merger._reject_ungrounded_prose = lambda *a, **k: []
        common.ungrounded_figures = lambda *a, **k: []
        return self

    def __exit__(self, *exc: object) -> None:
        from ceynex.agents import common
        from ceynex.orchestrator import merger

        merger._reject_ungrounded_prose, common.ungrounded_figures = self._saved


# --------------------------------------------------------------------- runners
async def run_ceynex(question: str, *, use_llm: bool) -> dict[str, Any]:
    from ceynex.orchestrator.demo import answer

    with GuardRecords() as guards:
        started = time.perf_counter()
        try:
            result = await answer(question, use_llm=use_llm, user_id="eval-baseline")
            error = None
        except Exception as exc:  # noqa: BLE001 - a crash is a result
            result, error = {}, str(exc)
        elapsed = (time.perf_counter() - started) * 1000
    return {
        "answer": result.get("answer", "") or "",
        "evidence": [str(e.get("claim", "")) + " " + str(e.get("detail", "")) for e in result.get("evidence", []) or []],
        "findings": [str(t) for t in result.get("findings", []) or []],
        "refused": is_refusal(result.get("answer", "") or "", result) if result else False,
        "degraded": bool(result.get("degraded", False)),
        "route": list(result.get("route", []) or []),
        "guard_discards": guards.discarded,
        "provider_gave_up": guards.gave_up,
        "elapsed_ms": elapsed,
        "error": error,
    }


def _openai_chat(question: str) -> tuple[str, dict[str, Any]]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = json.dumps({
        "model": A_MODEL,
        "temperature": A_TEMPERATURE,
        "max_tokens": 400,
        "messages": [{"role": "system", "content": A_SYSTEM}, {"role": "user", "content": question}],
    }).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URL
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"] or "", payload.get("usage", {})


async def run_llm_only(question: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        text, usage = await asyncio.to_thread(_openai_chat, question)
        error = None
    except Exception as exc:  # noqa: BLE001
        text, usage, error = "", {}, str(exc)
    return {
        "answer": text,
        "evidence": [],
        "findings": [],
        # No `unanswered` field exists for a bare model, so the harness falls back
        # to its refusal-marker list. Read these three answers by hand as well.
        "refused": is_refusal(text, {}) if text else True,
        "degraded": False,
        "route": [],
        "guard_discards": [],
        "provider_gave_up": 0,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "usage": usage,
        "error": error,
    }


# --------------------------------------------------------------------- scoring
def figures(text: str) -> list[str]:
    """Every non-structural figure in `text` (an empty corpus grounds nothing)."""
    return grounding.ungrounded_figures(text or "", [], direction_aware=False)


def score(run: dict[str, Any], reference: list[str], dataset: list[str] | None = None) -> dict[str, Any]:
    """`unsupported`: not in any evidence record (strict, what the evidence panel
    shows). `unsupported_dataset`: not in the evidence nor in any figure the
    agents computed from the data (what the grounding guard checks against)."""
    stated = figures(run["answer"])
    unsupported = grounding.ungrounded_figures(run["answer"] or "", reference)
    unsupported_dataset = grounding.ungrounded_figures(run["answer"] or "", dataset if dataset is not None else reference)
    return {"figures": stated, "unsupported": unsupported, "unsupported_dataset": unsupported_dataset}


async def one_repeat(questions: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    clear_cache()
    rows: dict[str, dict[str, Any]] = {q["id"]: {"question": q, "runs": {}} for q in questions}
    for condition in CONDITIONS:
        for q in questions:
            text = q["question"]
            if condition == "A":
                run = await run_llm_only(text)
            elif condition == "B":
                with GuardsOff():
                    run = await run_ceynex(text, use_llm=True)
            else:
                run = await run_ceynex(text, use_llm=condition == "C")
            rows[q["id"]]["runs"][condition] = run
            print(f"  repeat {index} {condition} {q['id']}: {run['elapsed_ms']:.0f} ms"
                  + (f" ERROR {run['error']}" if run["error"] else ""), flush=True)

    out = []
    for qid, row in rows.items():
        q = row["question"]
        reference = [q["question"]]
        for c in ("B", "C", "D"):
            reference += row["runs"][c]["evidence"]
        dataset = list(reference)
        for c in ("B", "C", "D"):
            dataset += row["runs"][c]["findings"]
        record = {"repeat": index, "id": qid, "category": q["category"], "question": q["question"],
                  "answerable": bool(q.get("answerable", True)), "reference_size": len(reference) - 1,
                  # Kept so eval/claims.py can judge claims against exactly what was scored.
                  "reference": sorted(set(reference[1:])),
                  "dataset_reference": sorted(set(dataset[1:]))}
        for c in CONDITIONS:
            run = row["runs"][c]
            record[c] = {**run, **score(run, reference, dataset)}
            record[c].pop("evidence", None)
            record[c].pop("findings", None)
        out.append(record)
    return out


# --------------------------------------------------------------------- report
def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    repeats = sorted({r["repeat"] for r in records})
    summary: dict[str, Any] = {"repeats": len(repeats), "questions": len({r["id"] for r in records})}
    for c in CONDITIONS:
        per_repeat = []
        for rep in repeats:
            rows = [r for r in records if r["repeat"] == rep]
            answerable = [r for r in rows if r["answerable"]]
            unanswerable = [r for r in rows if not r["answerable"]]
            total_figs = sum(len(r[c]["figures"]) for r in answerable)
            total_unsup = sum(len(r[c]["unsupported"]) for r in answerable)
            total_unsup_ds = sum(len(r[c].get("unsupported_dataset", r[c]["unsupported"])) for r in answerable)
            with_figs = [r for r in answerable if r[c]["figures"]]
            clean = [r for r in with_figs if not r[c]["unsupported"]]
            latencies = sorted(r[c]["elapsed_ms"] for r in rows if not r[c]["error"])
            per_repeat.append({
                "figures_per_answer": total_figs / len(answerable) if answerable else 0.0,
                "unsupported_figure_rate": total_unsup / total_figs if total_figs else 0.0,
                "answers_with_unsupported": sum(1 for r in answerable if r[c]["unsupported"]) / len(answerable),
                "unsupported_rate_dataset": total_unsup_ds / total_figs if total_figs else 0.0,
                "answers_with_unsupported_dataset": sum(
                    1 for r in answerable if r[c].get("unsupported_dataset", r[c]["unsupported"])) / len(answerable),
                "fully_supported_of_answers_with_figures": len(clean) / len(with_figs) if with_figs else 0.0,
                "answers_with_figures": len(with_figs) / len(answerable),
                "unanswerable_refused": (sum(1 for r in unanswerable if r[c]["refused"]) / len(unanswerable)) if unanswerable else 0.0,
                "crashed": sum(1 for r in rows if r[c]["error"]),
                "p50_ms": statistics.median(latencies) if latencies else 0.0,
                "p95_ms": latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))] if latencies else 0.0,
                "guard_interventions": sum(1 for r in rows if r[c]["guard_discards"]),
            })
        summary[c] = {k: _stats([p[k] for p in per_repeat]) for k in per_repeat[0]}
    return summary


def _stats(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def render(summary: dict[str, Any]) -> str:
    metrics = [
        ("figures_per_answer", "Figures per answerable answer", False),
        ("unsupported_rate_dataset", "Figures not from the dataset / all figures", True),
        ("answers_with_unsupported_dataset", "Answers with >= 1 figure not from the dataset", True),
        ("unsupported_figure_rate", "Figures not in a cited evidence record (strict)", True),
        ("answers_with_unsupported", "Answers with >= 1 figure not in a cited record (strict)", True),
        ("fully_supported_of_answers_with_figures", "Fully supported (answers with figures)", True),
        ("answers_with_figures", "Answerable answers that state a figure", True),
        ("unanswerable_refused", "Unanswerable questions refused", True),
        ("guard_interventions", "Grounding guard intervened (questions)", False),
        ("crashed", "Crashed", False),
        ("p50_ms", "Latency p50 (ms)", False),
        ("p95_ms", "Latency p95 (ms)", False),
    ]
    lines = [f"Baseline comparison: {summary['questions']} questions x {summary['repeats']} repeat(s); mean (sd)", ""]
    lines.append("| Metric | " + " | ".join(f"{c}: {LABELS[c]}" for c in "ABCD") + " |")
    lines.append("|---|" + "---|" * 4)
    for key, label, pct in metrics:
        cells = []
        for c in "ABCD":
            s = summary[c][key]
            cells.append(f"{s['mean']*100:.1f}% ({s['sd']*100:.1f})" if pct else f"{s['mean']:.1f} ({s['sd']:.1f})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare CeyNex with an LLM-only baseline and ablations.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--id", help="run a single question by id")
    parser.add_argument("--out", type=Path, default=Path("eval/results/baseline"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    isolate()
    questions = load_questions()
    if args.id:
        questions = [q for q in questions if q["id"] == args.id]
    args.out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index in range(1, args.repeat + 1):
        batch = asyncio.run(one_repeat(questions, index))
        records += batch
        (args.out / f"repeat_{index}.json").write_text(json.dumps(batch, indent=2, default=str))

    summary = summarise(records)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    table = render(summary)
    (args.out / "summary.md").write_text(table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
