"""PandaLM demo: pairwise judging on three scenarios.

We load PandaLM-7B with raw HuggingFace transformers (no extra package
needed) and apply PandaLM's official prompt template ourselves. This is
both clearer to read and lets you swap in your own prompt variant if you
want to study judge sensitivity to formatting.

Three scenarios in order:

  1. Clear quality gap (complete vs one-word answer)        → expect "1"
  2. Subtle factual error in response 2                      → expect "1"
  3. Two correct paraphrases of the same answer              → expect "Tie"

For each case we also evaluate the pair *reversed* (B then A) to surface
position bias — a known issue with all LLM judges.
"""
from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# The exact prompt template from the PandaLM repo (slightly reformatted for
# readability). Changing this even slightly degrades judging quality, since
# the model was trained on this exact format.
PROMPT_TEMPLATE = """Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.

### Instruction:
{instruction}

### Input:
{input}

### Response 1:
{response1}

### Response 2:
{response2}

### Evaluation:
"""


SCENARIOS = [
    # ----- 1. Clear quality gap --------------------------------------------
    {
        "name": "Clear quality gap",
        "instruction": "Explain why ice floats on water.",
        "input": "",
        "response_a": (
            "When water freezes, its molecules arrange themselves into a "
            "hexagonal crystal lattice that takes up more space per molecule "
            "than liquid water. Because the same mass of ice occupies a larger "
            "volume than liquid water, ice has a lower density and floats."
        ),
        "response_b": "Because it's lighter.",
        "expected": "1",
    },
    # ----- 2. Subtle factual error (one number wrong) ----------------------
    {
        "name": "Subtle factual error",
        "instruction": "How tall is Mount Everest, and where is it?",
        "input": "",
        "response_a": (
            "Mount Everest is about 8,849 metres (29,032 ft) tall and sits on "
            "the border between Nepal and the Tibet Autonomous Region of China."
        ),
        "response_b": (
            "Mount Everest is about 7,849 metres tall and sits on the border "
            "between Nepal and the Tibet Autonomous Region of China."
        ),
        "expected": "1",
    },
    # ----- 3. Equivalent quality (should ideally Tie) ----------------------
    {
        "name": "Equivalent paraphrases",
        "instruction": "What's the capital of Australia?",
        "input": "",
        "response_a": "The capital of Australia is Canberra.",
        "response_b": "Canberra is the capital of Australia.",
        "expected": "Tie",
    },
]


def load_model(name: str, device: str):
    print(f"Loading {name} (this is a ~14 GB download on first run)...")
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return model, tok


def judge(model, tok, instruction: str, input_: str, r1: str, r2: str,
          max_new_tokens: int = 256) -> dict:
    """One judging call; returns parsed {winner, rationale, reference}."""
    prompt = PROMPT_TEMPLATE.format(
        instruction=instruction, input=input_ or "Noinput.",
        response1=r1, response2=r2,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # judges should be deterministic
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_pandalm_output(generated)


def parse_pandalm_output(text: str) -> dict:
    """PandaLM's training format yields:

        <winner: 1|2|Tie>
        ### Reason: <one-paragraph rationale>
        ### Reference: <ideal answer the judge would write>
    """
    text = text.strip()
    winner_match = re.match(r"\s*(1|2|Tie)\b", text, re.IGNORECASE)
    winner = winner_match.group(1).capitalize() if winner_match else "?"

    reason_match = re.search(r"###\s*Reason:?\s*(.*?)(?=###|\Z)", text, re.DOTALL | re.IGNORECASE)
    reference_match = re.search(r"###\s*Reference:?\s*(.*?)\Z", text, re.DOTALL | re.IGNORECASE)
    return {
        "winner": winner,
        "rationale": (reason_match.group(1).strip() if reason_match else "").strip(),
        "reference": (reference_match.group(1).strip() if reference_match else "").strip(),
        "raw": text,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="WeOpenML/PandaLM-7B-v1")
    p.add_argument("--device", choices=("cuda", "cpu"),
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, tok = load_model(args.model, args.device)

    for scenario in SCENARIOS:
        print("\n" + "=" * 80)
        print(f"SCENARIO: {scenario['name']}    (expected winner: {scenario['expected']})")
        print("=" * 80)
        print(f"Instruction : {scenario['instruction']}")
        print(f"Response 1  : {scenario['response_a']}")
        print(f"Response 2  : {scenario['response_b']}\n")

        # Forward order: (A, B)
        out_ab = judge(
            model, tok,
            scenario["instruction"], scenario["input"],
            scenario["response_a"], scenario["response_b"],
        )
        # Reversed order: (B, A) — position-bias check.
        out_ba = judge(
            model, tok,
            scenario["instruction"], scenario["input"],
            scenario["response_b"], scenario["response_a"],
        )

        print(f"Judge (A as 1, B as 2):  winner = {out_ab['winner']}")
        print(f"  rationale: {out_ab['rationale'][:200]}")
        print(f"Judge (B as 1, A as 2):  winner = {out_ba['winner']}")
        print(f"  rationale: {out_ba['rationale'][:200]}")

        # A consistent judge picks the same response under both orderings:
        # winner_ab='1' means A wins; winner_ba='2' also means A wins.
        consistent = (
            (out_ab["winner"] == "1" and out_ba["winner"] == "2")
            or (out_ab["winner"] == "2" and out_ba["winner"] == "1")
            or (out_ab["winner"] == "Tie" and out_ba["winner"] == "Tie")
        )
        print(f"Position-consistent: {'YES' if consistent else 'NO (position bias detected)'}")


if __name__ == "__main__":
    main()
