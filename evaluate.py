"""Run the CQS judge over a dataset and report quality + calibration vs human labels.

Examples:
  python evaluate.py --mock                      # no API key needed
  python evaluate.py --model claude-sonnet-4-6   # real judge; needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import json
import math
import os

from cqs_judge import cqs_from_scores, get_judge


def pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx and sy else float("nan")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Score conversations with the CQS judge.")
    ap.add_argument("--data", default=os.path.join(here, "data", "sample_conversations.json"))
    ap.add_argument("--mock", action="store_true", help="use the no-API heuristic baseline")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--threshold", type=float, default=70.0, help="CQS pass/fail threshold")
    args = ap.parse_args()

    with open(args.data) as f:
        convs = json.load(f)

    judge = get_judge(mock=args.mock, model=args.model)
    print(f"\nJudge: {judge.model}   |   {len(convs)} conversations   |   pass threshold: {args.threshold:.0f}\n")
    print(f"{'id':<10}{'domain':<18}{'judge':>7}{'human':>7}{'gap':>7}  rationale")
    print("-" * 96)

    judge_cqs, human_cqs = [], []
    for c in convs:
        j = judge.score(c["turns"])
        jc, hc = j.cqs(), cqs_from_scores(c["human_label"])
        judge_cqs.append(jc)
        human_cqs.append(hc)
        print(f"{c['id']:<10}{c['domain']:<18}{jc:>7.1f}{hc:>7.1f}{jc - hc:>+7.1f}  {j.rationale[:42]}")

    n = len(convs)
    mae = sum(abs(a - b) for a, b in zip(judge_cqs, human_cqs)) / n
    within10 = sum(1 for a, b in zip(judge_cqs, human_cqs) if abs(a - b) <= 10) / n * 100
    passes = lambda v: v >= args.threshold
    agreement = sum(1 for a, b in zip(judge_cqs, human_cqs) if passes(a) == passes(b)) / n * 100
    r = pearson(judge_cqs, human_cqs)

    print("-" * 96)
    print(f"\nAggregate CQS  -  judge: {sum(judge_cqs) / n:5.1f}   human: {sum(human_cqs) / n:5.1f}\n")
    print("Calibration vs human labels")
    print(f"  Mean absolute error:   {mae:5.1f} pts")
    print(f"  Within +/- 10 pts:     {within10:5.0f}%")
    print(f"  Pass/fail agreement:   {agreement:5.0f}%")
    print(f"  Pearson correlation:   {r:5.2f}")
    print("\n(Run with --mock you are seeing the heuristic baseline; a real LLM judge")
    print(" calibrates far more tightly to human labels, especially on correctness.)\n")


if __name__ == "__main__":
    main()
