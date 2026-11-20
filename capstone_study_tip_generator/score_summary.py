"""Combine raw outputs and manual scores into the evaluation record and per-model means.
Run from this folder:  python score_summary.py   (writes lab_outputs/eval_summary.md)
"""
import json

outputs = {r["topic"]: r for r in json.load(open("lab_outputs/outputs.json", encoding="utf-8"))}
scores = json.load(open("data/scores.json", encoding="utf-8"))
assert set(outputs) == set(scores), "topic mismatch"

lines = ["| Topic | Model | Relevance | Readability | Usefulness | Total /6 | Notes |",
         "|---|---|---|---|---|---|---|"]
tot = {"base": [], "adapted": []}
crit = {"base": [[], [], []], "adapted": [[], [], []]}
for topic in outputs:
    for model in ("base", "adapted"):
        r, d, u, note = scores[topic][model]
        tot[model].append(r + d + u)
        for i, v in enumerate((r, d, u)):
            crit[model][i].append(v)
        lines.append(f"| {topic} | {model.capitalize()} | {r} | {d} | {u} | {r + d + u} | {note} |")

lines += ["", "| Model | Mean relevance | Mean readability | Mean usefulness | Mean total /6 |", "|---|---|---|---|---|"]
for model in ("base", "adapted"):
    m = [sum(c) / len(c) for c in crit[model]]
    lines.append(f"| {model.capitalize()} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} | {sum(tot[model]) / len(tot[model]):.2f} |")
wins = sum(a > b for a, b in zip(tot["adapted"], tot["base"]))
ties = sum(a == b for a, b in zip(tot["adapted"], tot["base"]))
lines.append(f"\nAdapted scored higher on {wins} topics, tied on {ties}, lower on {len(outputs) - wins - ties}.")
text = "\n".join(lines)
open("lab_outputs/eval_summary.md", "w", encoding="utf-8").write(text)
print(text)
