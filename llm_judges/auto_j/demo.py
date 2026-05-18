"""Auto-J demo: single-response scoring + pairwise critique, with optional 4-bit.

We load `GAIR/autoj-13b` (or whatever `--model` you pass), apply Auto-J's
official prompt templates, and run two evaluation modes against the same
scenario:

  1. SINGLE   - score one response 1..10, with a free-form critique.
  2. PAIRWISE - pick a winner between two responses, with critique.

`--quantization 4bit` loads via bitsandbytes NF4, which fits the 13B model
on a single 24 GB consumer GPU. Note that bitsandbytes requires CUDA;
on CPU / Apple Silicon, drop the flag and use fp16 / fp32.
"""
from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Auto-J's official prompt templates from the paper repo, slightly tightened.
# Changing these formats degrades judging quality; the model was trained on
# this exact wording.
SINGLE_PROMPT = """Write critiques for a submitted response on a given user's query, and grade the response.

[BEGIN DATA]
***
[Query]: {query}
***
[Response]: {response}
***
[END DATA]

Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"."""

PAIRWISE_PROMPT = """You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {query}
***
[Response 1]: {response1}
***
[Response 2]: {response2}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided."""


# Tricky factual question — both responses are fluent; one is correct.
QUERY = "Was the Treaty of Versailles signed before or after the Russian Revolution, and roughly how far apart were they?"

RESPONSE_GOOD = (
    "The Russian Revolution happened in 1917 (the February and October revolutions), and "
    "the Treaty of Versailles was signed on 28 June 1919. So the Treaty came after the "
    "Russian Revolution, by roughly a year and a half to two years."
)
RESPONSE_FLAWED = (
    "The Treaty of Versailles was signed in 1917, the same year as the Russian Revolution, "
    "so they happened essentially simultaneously. The two events are sometimes considered the "
    "twin pillars of the post-WWI international order."
)


def load_model(model_name: str, quantization: str):
    """Return (model, tokenizer). `quantization` ∈ {'none', '4bit', '8bit'}."""
    tok = AutoTokenizer.from_pretrained(model_name)

    kwargs: dict = {}
    if quantization in ("4bit", "8bit"):
        if not torch.cuda.is_available():
            raise SystemExit(
                f"--quantization {quantization} needs CUDA (bitsandbytes has no CPU "
                f"path). Drop the flag and use fp16/fp32, or run on a CUDA GPU."
            )
        from transformers import BitsAndBytesConfig

        # NF4 with double quantization is the recommended config: NF4 weight
        # format + a tiny quantization of the *quantization constants*. Saves
        # an extra ~0.5 bit per weight at near-zero quality cost.
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:  # 8bit
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

    print(f"Loading {model_name}  quantization={quantization}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tok


def generate(model, tok, prompt: str, max_new_tokens: int = 1024) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_single_rating(text: str) -> int | None:
    """Auto-J's single mode ends with `Rating: [[N]]`."""
    m = re.search(r"Rating:\s*\[\[(\d+)\]\]", text)
    return int(m.group(1)) if m else None


def parse_pairwise_decision(text: str) -> str:
    """Auto-J's pairwise mode ends with a line containing the final decision."""
    m = re.search(
        r"final decision is\s+(Response\s*1|Response\s*2|Tie)",
        text, re.IGNORECASE,
    )
    if not m:
        return "?"
    return m.group(1).strip().replace(" ", " ").title()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="GAIR/autoj-13b",
                   help="GAIR/autoj-13b (13B, English) or GAIR/autoj-bilingual-6b")
    p.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="none",
                   help="bitsandbytes quantization (CUDA only). 4bit fits 13B on a 24 GB GPU.")
    args = p.parse_args()

    model, tok = load_model(args.model, args.quantization)

    # --- 1. SINGLE-RESPONSE EVALUATION -----------------------------------
    print("\n" + "=" * 80)
    print("SINGLE-RESPONSE EVALUATION — score 1..10 + critique")
    print("=" * 80)
    for label, resp in (("GOOD", RESPONSE_GOOD), ("FLAWED", RESPONSE_FLAWED)):
        prompt = SINGLE_PROMPT.format(query=QUERY, response=resp)
        out = generate(model, tok, prompt)
        rating = parse_single_rating(out)
        print(f"\n--- Response: {label} ---")
        print(f"Rating: {rating if rating is not None else 'UNPARSEABLE'}/10")
        print(f"Critique:\n{out.strip()}\n")

    # --- 2. PAIRWISE EVALUATION ------------------------------------------
    print("\n" + "=" * 80)
    print("PAIRWISE EVALUATION — direct comparison + critique")
    print("=" * 80)
    print("Response 1 = GOOD, Response 2 = FLAWED  (a useful judge should pick Response 1)")
    prompt = PAIRWISE_PROMPT.format(
        query=QUERY, response1=RESPONSE_GOOD, response2=RESPONSE_FLAWED,
    )
    out = generate(model, tok, prompt)
    print(f"\nDecision: {parse_pairwise_decision(out)}")
    print(f"Full critique:\n{out.strip()}")


if __name__ == "__main__":
    main()
