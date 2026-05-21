"""Run two small Qwen models on three LongBench summarization subsets,
score with ROUGE-L and BERTScore, dump everything to results/results.json.

Top-to-bottom shape:
    1. Load N examples from each LongBench subset.
    2. For each model:
         load → for each subset: prompt → generate → record.
         free GPU memory before the next model.
    3. After all generations: score every (pred, ref) pair with ROUGE-L
       and BERTScore-F1, attach per-example and per-subset means.
    4. Write results/results.json.

Run:
    uv run python scripts/run.py --num 10
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    # Qwen3 small dense models. There is no "Qwen3.5" release; 0.6B / 1.7B are
    # the smallest dense checkpoints in the Qwen3 family. They support
    # thinking mode by default — we disable it below via the chat template
    # because we want fast direct summaries, not chain-of-thought traces.
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
]


# Reference numbers per (subset, metric) for context. Sources:
#
# - LongBench v1 paper (Bai et al., arXiv:2308.14508), Table 4: ROUGE-L F1
#   for closed and open LLMs on the same three subsets. Ranges quoted below
#   span "strong open ~7B in 32k-context tuning" to "GPT-4 / Claude-2".
# - For supervised SoTA (a *fine-tuned* abstractive model on the standalone
#   benchmark), we cite the published Pegasus / PRIMERA / LED results.
# - BERTScore-F1 numbers are with `rescale_with_baseline=False` (matches our
#   eval). Unrelated English text floors around 0.83; strong summarizers on
#   these corpora typically land in 0.85–0.88.
REFERENCES = {
    "multi_news": {
        "rougeL": {
            "frontier_llm": 0.264,     # GPT-3.5-Turbo-16k, LongBench paper
            "supervised_sota": 0.249,  # PRIMERA, Xiao et al. 2022
        },
        "bertscore_f1": {
            "supervised_sota": 0.866,
            "unrelated_floor": 0.83,
        },
    },
    "gov_report": {
        "rougeL": {
            "frontier_llm": 0.295,     # GPT-3.5-Turbo-16k, LongBench paper
            "supervised_sota": 0.351,  # LED / PEGASUS-X fine-tuned
        },
        "bertscore_f1": {
            "supervised_sota": 0.872,
            "unrelated_floor": 0.83,
        },
    },
    "qmsum": {
        "rougeL": {
            "frontier_llm": 0.234,     # GPT-3.5-Turbo-16k, LongBench paper
            "supervised_sota": 0.318,  # DialogLED / Locator+Summarizer
        },
        "bertscore_f1": {
            "supervised_sota": 0.862,
            "unrelated_floor": 0.83,
        },
    },
}

# Three English LongBench v1 summarization subsets. Each has a different
# input shape, hence the slightly different prompts.
SUBSETS: dict[str, dict] = {
    "multi_news": {
        "prompt_tmpl": (
            "You are a careful summarizer. Summarize the following news "
            "articles into a concise paragraph that covers the key facts.\n\n"
            "{context}\n\nSummary:"
        ),
        "needs_query": False,
    },
    "gov_report": {
        "prompt_tmpl": (
            "You are a careful summarizer. Summarize the following government "
            "report into a concise paragraph covering its main findings and "
            "recommendations.\n\n{context}\n\nSummary:"
        ),
        "needs_query": False,
    },
    "qmsum": {
        "prompt_tmpl": (
            "You are summarizing a meeting transcript to answer a specific "
            "query. Read the transcript, then give a concise answer to the "
            "query using only information from the transcript.\n\n"
            "Query: {query}\n\nTranscript:\n{context}\n\nAnswer:"
        ),
        "needs_query": True,
    },
}

# Truncate inputs by characters (not tokens) for speed of the truncation
# itself. ~12k chars ≈ ~3k tokens, well inside Qwen2.5's 32k context.
MAX_INPUT_CHARS = 12000


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_prompt(subset: str, example: dict) -> str:
    cfg = SUBSETS[subset]
    context = (example.get("context") or example.get("input") or "")[:MAX_INPUT_CHARS]
    if cfg["needs_query"]:
        return cfg["prompt_tmpl"].format(context=context, query=example.get("input", ""))
    return cfg["prompt_tmpl"].format(context=context)


def generate_for_model(model_id: str, data: dict[str, list[dict]],
                       max_new_tokens: int, device: str) -> dict[str, list[dict]]:
    print(f"\n→ Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    # Qwen3 chat template supports an `enable_thinking` flag that toggles
    # the model's chain-of-thought trace. We want clean summaries, not
    # reasoning blocks, so turn it off for any Qwen3 checkpoint.
    chat_kwargs: dict = {}
    if "qwen3" in model_id.lower():
        chat_kwargs["enable_thinking"] = False

    out: dict[str, list[dict]] = {}
    for subset, examples in data.items():
        rows = []
        for i, ex in enumerate(examples):
            user_msg = build_prompt(subset, ex)
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                return_tensors="pt",
                add_generation_prompt=True,
                **chat_kwargs,
            ).to(device)

            t0 = time.time()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.time() - t0

            pred = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            # Belt-and-braces: if a thinking-mode model leaked a <think>...
            # block (e.g. enable_thinking was ignored), strip it before
            # scoring so the metrics aren't dominated by reasoning text.
            pred = re.sub(r"<think>.*?</think>", "", pred, flags=re.DOTALL).strip()
            ref = (ex.get("answers") or [""])[0]
            ctx = (ex.get("context") or ex.get("input") or "")[:MAX_INPUT_CHARS]

            rows.append({
                "id": ex.get("_id", f"{subset}_{i}"),
                "context_preview": ctx[:600] + (" ..." if len(ctx) > 600 else ""),
                "context_chars": len(ctx),
                "query": ex.get("input", "") if SUBSETS[subset]["needs_query"] else "",
                "reference": ref,
                "prediction": pred,
                "gen_seconds": round(elapsed, 2),
                "input_tokens": int(input_ids.shape[1]),
            })
            print(f"   {subset}[{i + 1:>2}/{len(examples)}] "
                  f"{elapsed:5.1f}s  in_tok={input_ids.shape[1]:>5}  "
                  f"out_chars={len(pred)}")
        out[subset] = rows

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score_all(predictions: dict[str, dict[str, list[dict]]]) -> dict:
    """Attach per-example ROUGE-L + BERTScore-F1, return per-(model,subset) means."""
    from bert_score import score as bert_score_fn

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    aggregate: dict[str, dict[str, dict]] = {}
    for model_id, by_subset in predictions.items():
        aggregate[model_id] = {}
        for subset, rows in by_subset.items():
            preds = [r["prediction"] for r in rows]
            refs = [r["reference"] for r in rows]

            rouge_scores = [
                rouge.score(ref, pred)["rougeL"].fmeasure
                for pred, ref in zip(preds, refs)
            ]

            print(f"   BERTScore: {model_id} / {subset} ({len(preds)} pairs) ...")
            P, R, F = bert_score_fn(preds, refs, lang="en", verbose=False,
                                     rescale_with_baseline=False)
            f1_list = F.tolist()

            for r, rl, bf in zip(rows, rouge_scores, f1_list):
                r["rougeL"] = round(rl, 4)
                r["bertscore_f1"] = round(bf, 4)

            mean_rouge = sum(rouge_scores) / max(1, len(rouge_scores))
            mean_bert = sum(f1_list) / max(1, len(f1_list))
            mean_time = sum(r["gen_seconds"] for r in rows) / max(1, len(rows))

            aggregate[model_id][subset] = {
                "n": len(rows),
                "rougeL_mean": round(mean_rouge, 4),
                "bertscore_f1_mean": round(mean_bert, 4),
                "gen_seconds_mean": round(mean_time, 2),
            }
            print(f"     ROUGE-L={mean_rouge:.3f}  "
                  f"BERTScore-F1={mean_bert:.3f}  "
                  f"avg gen={mean_time:.1f}s")
    return aggregate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=10,
                   help="Examples per subset per model (default: 10).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output",
                   default=str(Path(__file__).resolve().parents[1] / "results" / "results.json"))
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # --- 1. Load N examples from each LongBench subset. ---
    # LongBench ships all subsets as JSONL files inside one data.zip on
    # HuggingFace. The modern `datasets` lib no longer runs the repo's
    # loading script, so we extract data.zip ourselves and read the JSONL.
    import zipfile

    cache_dir = Path.home() / ".cache" / "longbench-summarization"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not (cache_dir / "data").exists():
        print("Downloading LongBench data.zip ...")
        zip_path = hf_hub_download(repo_id="THUDM/LongBench",
                                   filename="data.zip", repo_type="dataset")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
        print(f"Extracted to {cache_dir / 'data'}")

    data: dict[str, list[dict]] = {}
    for subset in SUBSETS:
        path = cache_dir / "data" / f"{subset}.jsonl"
        with path.open() as f:
            rows = [json.loads(line) for line in f]
        examples = rows[:args.num]
        data[subset] = examples
        print(f"Loaded {len(examples)} examples from {path.name} "
              f"(of {len(rows)} total)")

    # --- 2. Generate with each model. ---
    predictions: dict[str, dict[str, list[dict]]] = {}
    for model_id in MODELS:
        predictions[model_id] = generate_for_model(
            model_id, data, args.max_new_tokens, device,
        )

    # --- 3. Score everything. ---
    print("\nScoring ...")
    aggregate = score_all(predictions)

    # --- 4. Dump JSON. ---
    payload = {
        "config": {
            "models": MODELS,
            "subsets": list(SUBSETS.keys()),
            "num_per_subset": args.num,
            "max_new_tokens": args.max_new_tokens,
            "max_input_chars": MAX_INPUT_CHARS,
            "device": device,
        },
        "references": REFERENCES,
        "aggregate": aggregate,
        "per_example": predictions,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
