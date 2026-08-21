"""Merge coherence — the one metric a machine cannot supply (M2 plan, Day 9).

SRS 3.1.2 forbids answers that concatenate per-agent responses. Nothing measures
that automatically: an answer reading "The agriculture agent says X. The apparel
agent says Y." is well-formed, correctly routed, fully grounded, and exactly the
failure the requirement is about. Only a reader can tell.

Two steps:

    python -m eval.coherence sheet  --out ratings.csv    # before the session
    python -m eval.coherence score  ratings.csv          # after it

**The sheet is blind in three ways.** Answers carry no agent attribution, no
question id, and appear in shuffled order, so a rater cannot infer how many
agents contributed and score the machinery instead of the prose. The mapping
back to question ids is written to a separate key file that raters do not see.

**Three raters, and the spread is reported with the mean.** Three raters
agreeing on 4 and three raters splitting 2/4/5 average the same and mean
completely different things. The spread is the honest part of the number.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RUBRIC = {
    5: "One answer. Reads as a single analysis; a reader could not tell how many agents contributed.",
    4: "Mostly unified. One or two seams, but the argument holds together.",
    3: "Stitched. Related points in sequence, no connective reasoning between them.",
    2: "Listed. Clearly separate findings placed next to each other.",
    1: "Concatenated. Reads as several answers pasted together, possibly contradicting.",
}

SEED = 20260819  # fixed so the shuffle is reproducible and the key always matches


@dataclass
class Sheet:
    rows: list[dict[str, Any]]
    key: dict[str, str]


def build_sheet(results: list[dict[str, Any]], *, seed: int = SEED) -> Sheet:
    """Blind, shuffled rating rows plus the key that maps them back."""
    scored = [r for r in results if (r.get("answer") or "").strip()]
    shuffled = list(scored)
    random.Random(seed).shuffle(shuffled)

    rows, key = [], {}
    for index, result in enumerate(shuffled, start=1):
        label = f"A{index:02d}"
        key[label] = result["id"]
        rows.append(
            {
                "label": label,
                "question": result["question"],
                "answer": " ".join((result.get("answer") or "").split()),
                "rating_1_to_5": "",
                "comment": "",
            }
        )
    return Sheet(rows=rows, key=key)


def write_sheet(sheet: Sheet, out: Path) -> None:
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet.rows[0]))
        writer.writeheader()
        writer.writerows(sheet.rows)

    key_path = out.with_suffix(".key.json")
    key_path.write_text(json.dumps(sheet.key, indent=2))

    rubric_path = out.with_suffix(".rubric.txt")
    rubric_path.write_text(
        "Merge coherence — rate each answer 1 to 5.\n\n"
        "You are rating whether the answer reads as ONE analysis, not whether it\n"
        "is correct, well-written, or complete. A wrong answer can be coherent\n"
        "and a right answer can be stitched together.\n\n"
        + "\n".join(f"  {score} — {text}" for score, text in sorted(RUBRIC.items(), reverse=True))
        + "\n\nFill the rating_1_to_5 column. Do not discuss with the other raters\n"
        "until all three sheets are done.\n"
    )
    print(f"wrote {out} ({len(sheet.rows)} answers), {key_path.name}, {rubric_path.name}")
    print("Give each rater their own copy of the CSV and the rubric. Never the key.")


def score(paths: list[Path], key: dict[str, str] | None = None) -> dict[str, Any]:
    """Aggregate one CSV per rater into a mean, a spread and per-answer detail."""
    per_rater: dict[str, dict[str, int]] = {}
    for path in paths:
        ratings = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                raw = (row.get("rating_1_to_5") or "").strip()
                if not raw:
                    continue
                value = int(float(raw))
                if not 1 <= value <= 5:
                    raise ValueError(f"{path.name}: rating {value} for {row['label']} is outside 1-5")
                ratings[row["label"]] = value
        if not ratings:
            raise ValueError(f"{path.name} has no ratings filled in")
        per_rater[path.stem] = ratings

    labels = sorted(set().union(*(set(r) for r in per_rater.values())))
    per_answer = []
    for label in labels:
        scores = [r[label] for r in per_rater.values() if label in r]
        per_answer.append(
            {
                "label": label,
                "question_id": (key or {}).get(label),
                "scores": scores,
                "mean": round(statistics.fmean(scores), 2),
                "spread": max(scores) - min(scores) if len(scores) > 1 else 0,
            }
        )

    everything = [s for entry in per_answer for s in entry["scores"]]
    disputed = [e for e in per_answer if e["spread"] >= 2]

    return {
        "raters": len(per_rater),
        "answers_rated": len(per_answer),
        "mean_coherence": round(statistics.fmean(everything), 2),
        "median_coherence": statistics.median(everything),
        "stdev": round(statistics.stdev(everything), 2) if len(everything) > 1 else 0.0,
        "mean_spread_between_raters": round(
            statistics.fmean([e["spread"] for e in per_answer]), 2
        ),
        "answers_rated_3_or_below": sum(1 for e in per_answer if e["mean"] <= 3.0),
        "disputed_answers": [e["label"] for e in disputed],
        "per_answer": per_answer,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blind merge-coherence rating.")
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("sheet", help="build blind rating sheets from a harness run")
    make.add_argument("--results", type=Path, required=True, help="JSON written by eval.harness --json")
    make.add_argument("--out", type=Path, default=Path("coherence_sheet.csv"))

    grade = sub.add_parser("score", help="aggregate the filled-in sheets")
    grade.add_argument("sheets", type=Path, nargs="+", help="one CSV per rater")
    grade.add_argument("--key", type=Path, help="the .key.json written alongside the sheet")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "sheet":
        payload = json.loads(args.results.read_text())
        sheet = build_sheet(payload["results"])
        if not sheet.rows:
            print("no answers to rate", file=sys.stderr)
            return 1
        write_sheet(sheet, args.out)
        return 0

    if len(args.sheets) < 3:
        # Not fatal: one rater is still a number. But it is a different claim,
        # and the report has to say so rather than quietly presenting n=1.
        print(
            f"warning: {len(args.sheets)} rater(s); the plan calls for 3. "
            "Record the count as a limitation in EVALUATION.md.",
            file=sys.stderr,
        )
    key = json.loads(args.key.read_text()) if args.key else None
    print(json.dumps(score(args.sheets, key), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
