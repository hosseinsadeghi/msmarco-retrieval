"""Render results/results.json as a self-contained HTML report.

Run:
    uv run python scripts/report.py
    uv run python scripts/report.py --input results/results.json --output results/report.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


CSS = """
:root {
  --bg: #fafafa;
  --fg: #1c1c1c;
  --muted: #6b6b6b;
  --border: #e2e2e2;
  --accent: #2b6cb0;
  --accent-light: #e6f0fa;
  --good: #2f855a;
  --bar1: #4c72b0;
  --bar2: #dd8452;
}
* { box-sizing: border-box; }
body {
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--fg);
  margin: 0; padding: 32px 48px 64px;
  max-width: 1200px; margin-left: auto; margin-right: auto;
}
h1 { margin: 0 0 8px; font-size: 26px; }
h2 { margin: 40px 0 12px; font-size: 20px;
     border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { margin: 24px 0 8px; font-size: 16px; color: var(--muted); }
.sub { color: var(--muted); margin-bottom: 24px; }
.config {
  background: white; border: 1px solid var(--border); border-radius: 6px;
  padding: 12px 16px; font-size: 13px; color: var(--muted);
  display: flex; flex-wrap: wrap; gap: 16px;
}
.config b { color: var(--fg); }
table { border-collapse: collapse; width: 100%; background: white;
        border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
th, td { padding: 10px 14px; text-align: left;
         border-bottom: 1px solid var(--border); }
th { background: #f3f4f6; font-weight: 600; font-size: 13px;
     letter-spacing: 0.02em; text-transform: uppercase; color: #555; }
tr:last-child td { border-bottom: none; }
td.num { font-variant-numeric: tabular-nums; }
td.best { background: var(--accent-light); font-weight: 600; color: var(--good); }
.bars { display: grid; grid-template-columns: 180px 1fr 80px;
        gap: 8px 12px; align-items: center; margin: 8px 0; }
.bar-row .label { color: var(--muted); font-size: 13px; }
.bar-track { background: #eef0f3; height: 14px; border-radius: 7px;
             overflow: hidden; position: relative; }
.bar-fill { height: 100%; border-radius: 7px; }
.bar-val { font-variant-numeric: tabular-nums; font-size: 13px;
           text-align: right; color: var(--muted); }
.card {
  background: white; border: 1px solid var(--border); border-radius: 8px;
  padding: 16px 20px; margin-bottom: 14px;
}
.card .meta {
  font-size: 12px; color: var(--muted); margin-bottom: 8px;
  display: flex; gap: 14px; flex-wrap: wrap;
}
.card .meta span b { color: var(--fg); font-weight: 600; }
.card .row { display: grid; grid-template-columns: 1fr 1fr 1fr;
             gap: 12px; margin-top: 8px; }
.card .row .col h4 { margin: 0 0 4px; font-size: 12px; color: var(--muted);
                     text-transform: uppercase; letter-spacing: 0.05em; }
.card .row .col .body { font-size: 13px; white-space: pre-wrap;
                        max-height: 220px; overflow-y: auto;
                        background: #fafbfc; border: 1px solid var(--border);
                        border-radius: 4px; padding: 8px 10px; }
.card details summary { cursor: pointer; font-size: 13px; color: var(--accent);
                        margin-top: 8px; }
.card details .full { font-size: 13px; white-space: pre-wrap; margin-top: 8px;
                      background: #fafbfc; border: 1px solid var(--border);
                      border-radius: 4px; padding: 10px 12px;
                      max-height: 420px; overflow-y: auto; }
.scores { display: flex; gap: 18px; font-variant-numeric: tabular-nums;
          font-size: 12px; }
.scores .s b { color: var(--fg); }
"""


def _esc(s: str) -> str:
    return html.escape(s or "", quote=False)


def _aggregate_table(agg: dict, models: list[str], subsets: list[str]) -> str:
    """One row per (model, subset)."""
    # Find best (max) score per metric per subset for highlighting.
    best = {}  # {(metric, subset): max_val}
    for m in models:
        for s in subsets:
            cell = agg[m][s]
            for metric in ("rougeL_mean", "bertscore_f1_mean"):
                key = (metric, s)
                best[key] = max(best.get(key, -1), cell[metric])

    rows = []
    for m in models:
        for s in subsets:
            cell = agg[m][s]
            rl = cell["rougeL_mean"]
            bf = cell["bertscore_f1_mean"]
            rl_cls = "num best" if rl == best[("rougeL_mean", s)] else "num"
            bf_cls = "num best" if bf == best[("bertscore_f1_mean", s)] else "num"
            rows.append(
                f"<tr><td>{_esc(m)}</td><td>{_esc(s)}</td>"
                f"<td class='num'>{cell['n']}</td>"
                f"<td class='{rl_cls}'>{rl:.3f}</td>"
                f"<td class='{bf_cls}'>{bf:.3f}</td>"
                f"<td class='num'>{cell['gen_seconds_mean']:.1f}</td></tr>"
            )
    return (
        "<table><thead><tr>"
        "<th>Model</th><th>Subset</th><th>N</th>"
        "<th>ROUGE-L</th><th>BERTScore-F1</th><th>Avg gen (s)</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def _bars(agg: dict, models: list[str], subsets: list[str]) -> str:
    """One bar chart per metric, with one row per (model, subset)."""
    out = []
    for metric, label, vmax in [
        ("rougeL_mean", "ROUGE-L", 0.5),
        ("bertscore_f1_mean", "BERTScore-F1", 1.0),
    ]:
        out.append(f"<h3>{label}</h3>")
        bar_color = "var(--bar1)" if metric == "rougeL_mean" else "var(--bar2)"
        # max across all cells for the scale (capped at vmax to keep tiny scores visible)
        cell_vals = [agg[m][s][metric] for m in models for s in subsets]
        scale = max(max(cell_vals) * 1.1, 0.05)
        scale = min(scale, vmax)
        for m in models:
            short = m.split("/")[-1]
            for s in subsets:
                v = agg[m][s][metric]
                w = max(2, min(100, v / scale * 100))
                out.append(
                    "<div class='bars bar-row'>"
                    f"<div class='label'>{_esc(short)} · {_esc(s)}</div>"
                    "<div class='bar-track'>"
                    f"<div class='bar-fill' style='width:{w:.1f}%; background:{bar_color};'></div>"
                    "</div>"
                    f"<div class='bar-val'>{v:.3f}</div>"
                    "</div>"
                )
    return "\n".join(out)


def _example_cards(per_example: dict, models: list[str], subsets: list[str],
                   k: int = 2) -> str:
    """Show k examples per subset: context preview + reference + each model's prediction + scores."""
    out = []
    for s in subsets:
        out.append(f"<h3>{_esc(s)}</h3>")
        # Use the first model's row list as the index source.
        ref_rows = per_example[models[0]][s]
        for i, row0 in enumerate(ref_rows[:k]):
            ref = row0["reference"]
            ctx_preview = row0["context_preview"]
            ctx_chars = row0["context_chars"]
            query = row0.get("query", "")

            # Pull the predictions for this index from each model.
            cols = []
            scores_lines = []
            for m in models:
                row = per_example[m][s][i]
                cols.append(
                    "<div class='col'><h4>"
                    f"{_esc(m.split('/')[-1])}</h4>"
                    f"<div class='body'>{_esc(row['prediction'])}</div></div>"
                )
                scores_lines.append(
                    f"<span class='s'>{_esc(m.split('/')[-1])}: "
                    f"<b>R-L</b> {row['rougeL']:.3f} · "
                    f"<b>BS-F1</b> {row['bertscore_f1']:.3f} · "
                    f"<b>{row['gen_seconds']:.1f}s</b></span>"
                )

            ref_col = (
                "<div class='col'><h4>Reference</h4>"
                f"<div class='body'>{_esc(ref)}</div></div>"
            )

            query_html = (
                f"<div class='meta'><span><b>Query:</b> {_esc(query)}</span></div>"
                if query else ""
            )
            out.append(
                "<div class='card'>"
                f"<div class='meta'>"
                f"<span><b>id:</b> {_esc(str(row0.get('id', '')))}</span>"
                f"<span><b>context:</b> {ctx_chars:,} chars</span>"
                f"</div>"
                f"{query_html}"
                f"<div class='scores'>{' '.join(scores_lines)}</div>"
                f"<div class='row'>{ref_col}{''.join(cols)}</div>"
                f"<details><summary>Show full context preview</summary>"
                f"<div class='full'>{_esc(ctx_preview)}</div></details>"
                "</div>"
            )
    return "\n".join(out)


def render(data: dict) -> str:
    cfg = data["config"]
    agg = data["aggregate"]
    per_example = data["per_example"]
    models = cfg["models"]
    subsets = cfg["subsets"]

    config_html = (
        f"<div class='config'>"
        f"<div><b>Models:</b> {', '.join(_esc(m) for m in models)}</div>"
        f"<div><b>Subsets:</b> {', '.join(_esc(s) for s in subsets)}</div>"
        f"<div><b>N per subset:</b> {cfg['num_per_subset']}</div>"
        f"<div><b>Max new tokens:</b> {cfg['max_new_tokens']}</div>"
        f"<div><b>Max input chars:</b> {cfg['max_input_chars']:,}</div>"
        f"<div><b>Device:</b> {_esc(cfg['device'])}</div>"
        f"</div>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Summarization benchmark — Qwen2.5 small models on LongBench</title>
<style>{CSS}</style>
</head>
<body>
<h1>Summarization benchmark</h1>
<p class="sub">Small Qwen2.5 instruction-tuned models on LongBench v1
summarization subsets, scored with ROUGE-L (lexical) and BERTScore-F1
(semantic).</p>

{config_html}

<h2>Aggregate scores</h2>
<p class="sub">Best score per (metric × subset) highlighted.</p>
{_aggregate_table(agg, models, subsets)}

<h2>Per-metric comparison</h2>
{_bars(agg, models, subsets)}

<h2>Example outputs</h2>
<p class="sub">Two examples per subset. Click <i>Show full context preview</i>
on any card to see the truncated input the models saw.</p>
{_example_cards(per_example, models, subsets, k=2)}

</body>
</html>
"""


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(here / "results" / "results.json"))
    p.add_argument("--output", default=str(here / "results" / "report.html"))
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text())
    html_out = render(data)
    Path(args.output).write_text(html_out)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
