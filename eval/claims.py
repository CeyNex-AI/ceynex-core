"""Claim-level check of the baseline comparison: are an answer's statements,
not only its numbers, consistent with the dataset?

    python -m eval.claims --runs eval/results/baseline            # judge every repeat
    python -m eval.claims --runs eval/results/baseline --agreement labels.csv

`eval.baseline` scores figures. An answer can be wrong without a figure: asked
which market takes the most Sri Lankan tea, the LLM-only baseline answered
"Russia" with no number at all, while the dataset's latest year says Iraq
(12.4%). This module catches that class of error.

**Method.** For each answerable question and each of conditions A, C and D
(B's prose differs from C's only when the guard fires, so it adds little here),
a judge model extracts up to six substantive trade claims from the answer and
labels each against the reference records that `eval.baseline` stored for that
question:

- SUPPORTED: the records state it, or it follows by simple arithmetic;
- CONTRADICTED: the records state something incompatible for the same item and
  period (another top market, the opposite direction, a different value);
- UNVERIFIABLE: the records neither support nor contradict it, including any
  claim about a period the records do not cover.

The judge works only from the records, never from its own knowledge, at
temperature 0 with a JSON response. An LLM judge has its own error, so a blind
sample is written for people to label (`audit_blind.csv`, with the condition and
the judge's label withheld; `audit_key.csv` holds them) and `--agreement`
reports Cohen's kappa between the people and the judge. The judge is the same
model family as condition A, which could favour A; any such bias would make A
look better, not worse.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any

JUDGE_MODEL = "gpt-4o"
JUDGED = ("A", "C", "D")
LABELS = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
MAX_REFERENCE_CHARS = 12_000
CONCURRENCY = 4

JUDGE_SYSTEM = """You are a careful fact-checker for trade statistics.
You receive a QUESTION, REFERENCE records taken from a trade dataset, and an ANSWER.

1. List the answer's substantive factual claims about trade, at most six: which market or product, rankings, the direction of a change (rose or fell), magnitudes, comparisons. Ignore hedges, advice, methodology, definitions, and statements about data availability or about the system itself.
2. Label each claim using ONLY the reference records, never your own knowledge:
   - SUPPORTED: the records state it, or it follows from them by simple arithmetic.
   - CONTRADICTED: the records state something incompatible for the same item and period, such as a different top market, the opposite direction, or a clearly different value.
   - UNVERIFIABLE: the records neither support nor contradict it. A claim about a period the records do not cover is UNVERIFIABLE. A claim presented as current or "latest" is compared with the latest period in the records.
3. For SUPPORTED or CONTRADICTED, quote the deciding record briefly.

Return JSON only: {"claims": [{"claim": "...", "label": "SUPPORTED|CONTRADICTED|UNVERIFIABLE", "reference_quote": "..."}]}.
If the answer makes no substantive trade claim, return {"claims": []}."""


def judge_prompt(question: str, reference: list[str], answer: str) -> str:
    records, used = [], 0
    for text in reference:
        if used + len(text) > MAX_REFERENCE_CHARS:
            records.append("[... further records omitted ...]")
            break
        records.append(f"- {text.strip()}")
        used += len(text)
    body = "\n".join(records) if records else "(no records)"
    return f"QUESTION:\n{question}\n\nREFERENCE records:\n{body}\n\nANSWER:\n{answer}"


def _call_judge(prompt: str) -> dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = json.dumps({
        "model": JUDGE_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
    }).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310 - fixed https URL
        payload = json.load(response)
    return parse_judgement(payload["choices"][0]["message"]["content"] or "")


def parse_judgement(content: str) -> dict[str, Any]:
    """The judge's JSON, with any label outside the three normalised away."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {"claims": [], "parse_error": content[:200]}
    claims = []
    for item in data.get("claims", []) or []:
        label = str(item.get("label", "")).upper().strip()
        if label not in LABELS:
            label = "UNVERIFIABLE"
        claims.append({"claim": str(item.get("claim", "")), "label": label,
                       "reference_quote": str(item.get("reference_quote", ""))})
    return {"claims": claims}


async def judge_all(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    jobs = []

    async def one(record: dict[str, Any], condition: str) -> dict[str, Any]:
        answer = record[condition]["answer"]
        async with semaphore:
            if not answer.strip():
                verdict = {"claims": []}
            else:
                try:
                    verdict = await asyncio.to_thread(
                        _call_judge, judge_prompt(record["question"], record.get("reference", []), answer))
                except Exception as exc:  # noqa: BLE001 - a failed judgement is recorded, not fatal
                    verdict = {"claims": [], "error": str(exc)}
        print(f"  judged repeat {record['repeat']} {record['id']} {condition}: "
              f"{len(verdict['claims'])} claim(s)", flush=True)
        return {"repeat": record["repeat"], "id": record["id"], "condition": condition,
                "question": record["question"], "answer": answer, **verdict}

    for record in records:
        if not record["answerable"]:
            continue
        for condition in JUDGED:
            jobs.append(one(record, condition))
    return await asyncio.gather(*jobs)


def summarise(judgements: list[dict[str, Any]]) -> dict[str, Any]:
    repeats = sorted({j["repeat"] for j in judgements})
    out: dict[str, Any] = {"repeats": len(repeats)}
    for condition in JUDGED:
        per_repeat = []
        for rep in repeats:
            rows = [j for j in judgements if j["condition"] == condition and j["repeat"] == rep]
            claims = [c for j in rows for c in j["claims"]]
            n = len(claims)
            per_repeat.append({
                "claims_per_answer": n / len(rows) if rows else 0.0,
                "supported": sum(c["label"] == "SUPPORTED" for c in claims) / n if n else 0.0,
                "contradicted": sum(c["label"] == "CONTRADICTED" for c in claims) / n if n else 0.0,
                "unverifiable": sum(c["label"] == "UNVERIFIABLE" for c in claims) / n if n else 0.0,
                "answers_with_contradiction": (sum(any(c["label"] == "CONTRADICTED" for c in j["claims"]) for j in rows)
                                               / len(rows)) if rows else 0.0,
                "judge_errors": sum(1 for j in rows if "error" in j or "parse_error" in j),
            })
        out[condition] = {k: {"mean": statistics.fmean(p[k] for p in per_repeat),
                              "sd": statistics.stdev([p[k] for p in per_repeat]) if len(per_repeat) > 1 else 0.0}
                          for k in per_repeat[0]}
    return out


def render(summary: dict[str, Any]) -> str:
    rows = [("claims_per_answer", "Claims per answer", False), ("supported", "Supported", True),
            ("contradicted", "Contradicted", True), ("unverifiable", "Unverifiable", True),
            ("answers_with_contradiction", "Answers with >= 1 contradiction", True),
            ("judge_errors", "Judge errors", False)]
    lines = [f"Claim-level check over {summary['repeats']} repeat(s); mean (sd)", "",
             "| Metric | " + " | ".join(JUDGED) + " |", "|---|" + "---|" * len(JUDGED)]
    for key, label, pct in rows:
        cells = [(f"{summary[c][key]['mean']*100:.1f}% ({summary[c][key]['sd']*100:.1f})" if pct
                  else f"{summary[c][key]['mean']:.2f} ({summary[c][key]['sd']:.2f})") for c in JUDGED]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_audit(judgements: list[dict[str, Any]], out: Path, size: int, seed: int = 7) -> int:
    """A blind sample, balanced across conditions, for people to label."""
    rng = random.Random(seed)
    pool = {c: [(j, claim) for j in judgements if j["condition"] == c for claim in j["claims"]] for c in JUDGED}
    sample = []
    for c in JUDGED:
        sample += rng.sample(pool[c], min(len(pool[c]), size // len(JUDGED)))
    rng.shuffle(sample)
    with (out / "audit_blind.csv").open("w", newline="", encoding="utf-8") as blind, \
         (out / "audit_key.csv").open("w", newline="", encoding="utf-8") as key:
        bw, kw = csv.writer(blind), csv.writer(key)
        bw.writerow(["item", "question", "claim", "human_label (SUPPORTED/CONTRADICTED/UNVERIFIABLE)"])
        kw.writerow(["item", "condition", "repeat", "id", "judge_label", "reference_quote"])
        for n, (j, claim) in enumerate(sample, 1):
            bw.writerow([n, j["question"], claim["claim"], ""])
            kw.writerow([n, j["condition"], j["repeat"], j["id"], claim["label"], claim["reference_quote"]])
    return len(sample)


def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in LABELS)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(labels_csv: Path, key_csv: Path) -> dict[str, Any]:
    with labels_csv.open(encoding="utf-8") as f:
        human = {row["item"]: row[next(k for k in row if k.startswith("human_label"))].strip().upper()
                 for row in csv.DictReader(f)}
    with key_csv.open(encoding="utf-8") as f:
        judge = {row["item"]: row["judge_label"] for row in csv.DictReader(f)}
    items = [i for i in human if human[i] in LABELS and i in judge]
    h, j = [human[i] for i in items], [judge[i] for i in items]
    return {"items": len(items), "agreement": sum(x == y for x, y in zip(h, j, strict=True)) / len(items) if items else 0.0,
            "kappa": cohens_kappa(h, j)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Claim-level check of eval.baseline results.")
    parser.add_argument("--runs", type=Path, default=Path("eval/results/baseline"))
    parser.add_argument("--audit-size", type=int, default=60)
    parser.add_argument("--agreement", type=Path, help="human-labelled audit_blind.csv")
    args = parser.parse_args(argv)

    if args.agreement:
        print(json.dumps(agreement(args.agreement, args.runs / "audit_key.csv"), indent=2))
        return 0

    records = []
    for path in sorted(args.runs.glob("repeat_*.json")):
        records += json.loads(path.read_text())
    if not records:
        print(f"no repeat_*.json under {args.runs}", file=sys.stderr)
        return 1
    if not any("reference" in r for r in records):
        print("these results predate eval.baseline storing its reference; rerun it", file=sys.stderr)
        return 1

    judgements = asyncio.run(judge_all(records))
    (args.runs / "claims.json").write_text(json.dumps(judgements, indent=2))
    summary = summarise(judgements)
    (args.runs / "claims_summary.json").write_text(json.dumps(summary, indent=2))
    table = render(summary)
    (args.runs / "claims_summary.md").write_text(table + "\n")
    sampled = write_audit(judgements, args.runs, args.audit_size)
    print(table)
    print(f"\nwrote a blind audit sample of {sampled} claims to {args.runs / 'audit_blind.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
