"""GAIA demo — weak browsing agent on the validation split, scored exact-match.

This is intentionally a **weak baseline**:
  - No JS-rendered pages (requests + bs4 only).
  - No file-type-aware reading (PDFs, spreadsheets, images attached to
    questions get text-extracted on a best-effort basis).
  - One web search per turn, max 5 turns.

Run:
    huggingface-cli login                # GAIA is a gated dataset
    uv run python gaia/demo.py --level 1 --num 3
"""
from __future__ import annotations

import argparse
import json
import os
import re

import requests
from bs4 import BeautifulSoup


def _fetch(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and return cleaned text (truncated)."""
    try:
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "ml-playground/0.1 (agent_eval)"})
        r.raise_for_status()
    except Exception as e:
        return f"[fetch error: {e}]"
    soup = BeautifulSoup(r.text, "lxml")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return text[:max_chars]


def _web_search(query: str, n: int = 3) -> list[dict]:
    """Stubbed — point this at SerpAPI / Bing / Brave for a real run."""
    return [{"title": f"(no-op search) {query}", "url": "", "snippet": ""}]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Returns a list of (title, url, snippet).",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL and return cleaned page text.",
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string"}}, "required": ["url"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer for grading. Use the most "
                           "concise form possible (a number, a short phrase).",
            "parameters": {"type": "object", "properties": {
                "answer": {"type": "string"}}, "required": ["answer"]},
        },
    },
]


def run_agent(question: str, model: str, max_turns: int = 5) -> str:
    from litellm import completion

    messages = [
        {"role": "system",
         "content": "You are a careful research assistant. Use the tools to "
                    "find the answer, then call submit_answer with the single "
                    "most concise correct answer."},
        {"role": "user", "content": question},
    ]
    for _ in range(max_turns):
        resp = completion(model=model, messages=messages, tools=TOOLS,
                          tool_choice="auto", temperature=0)
        msg = resp["choices"][0]["message"]
        messages.append(msg)
        if not msg.get("tool_calls"):
            return msg.get("content", "").strip()
        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            if name == "submit_answer":
                return args.get("answer", "").strip()
            elif name == "web_search":
                result = _web_search(args["query"])
            elif name == "fetch_url":
                result = _fetch(args["url"])
            else:
                result = f"unknown tool: {name}"
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result)})
    return ""


def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.strip().lower())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--num", type=int, default=3)
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY.")

    from datasets import load_dataset

    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
    ds = ds.filter(lambda r: r["Level"] == args.level).select(range(args.num))

    correct = 0
    for row in ds:
        pred = run_agent(row["Question"], args.model)
        ok = _normalize(pred) == _normalize(row["Final answer"])
        correct += int(ok)
        print(f"[{'OK' if ok else 'XX'}] {row['task_id']}: "
              f"pred={pred!r:.60}  gold={row['Final answer']!r:.60}")

    print(f"\nLevel {args.level}: {correct}/{len(ds)} correct "
          f"({correct/len(ds):.0%})")


if __name__ == "__main__":
    main()
