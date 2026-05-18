# RouteLLM — query routing between a strong & weak LLM

## What it is

[RouteLLM](https://github.com/lm-sys/RouteLLM) (Ong et al., LMSYS, 2024) is a
**router**: a small model that looks at an incoming prompt and predicts
whether you can answer it well with a cheap weak LLM, or whether you need to
escalate to an expensive strong LLM.

Trained on **preference data from Chatbot Arena** (which model did humans
prefer on which prompt?), it learns a real-valued score:

> `win_rate ∈ [0, 1]` — probability the *strong* model would beat the *weak*
> model on this prompt, according to learned human preferences.

You then pick a **threshold**: if `win_rate ≥ τ`, send to the strong model;
otherwise stay with the weak one. The threshold lets you trade off cost vs
quality without retraining.

Four pre-trained routers ship with the library, in increasing complexity /
accuracy:

| Router      | What it is                                       | Runs locally? |
| ---         | ---                                              | ---            |
| `random`    | Baseline: route at random with probability τ.    | yes |
| `mf`        | Matrix-factorization over Arena prompts.         | **no** — encodes prompts with `text-embedding-ada-002` via the OpenAI API |
| `sw_ranking`| Similarity-weighted ranking using embeddings.    | **no** — same OpenAI embedding dependency |
| `bert`      | BERT classifier fine-tuned on Arena preferences. | yes — fully local |
| `causal_llm`| LLaMA-3-8B fine-tuned as the router.             | yes — but needs ~16 GB VRAM |

Both `mf` and `sw_ranking` look small on paper but make a paid OpenAI API
call per prompt; in practice the only fully-local-and-small option is
`bert`. The demo defaults to `bert` for that reason — pass `--router mf`
if you have an `OPENAI_API_KEY` configured and want to try matrix
factorization.

The original paper showed that with the right router, you can recover
**95% of GPT-4-tier quality** on MT-Bench while using GPT-4 for only
**~25% of prompts** — a 4× cost cut on the OpenAI bill.

## Why this exists

In practice the prompt distribution in any LLM-backed product is wildly
non-uniform. A "summarize this email" prompt and a "prove this theorem"
prompt both go through the same chat box, but only one of them needs a
state-of-the-art model. Always-strong is wasteful; always-weak is a quality
disaster. Routing solves this without touching the underlying models.

The same architecture is also how OpenRouter-style multi-provider gateways
internally pick a model, and how production teams cut spend at scale on top
of providers like Anthropic / OpenAI.

## When to reach for it

- **Heterogeneous prompt mix** — a chatbot that gets both "what's 2+2" and
  "design a distributed consensus protocol". Big win.
- **Cost-sensitive production** — model usage is your dominant variable
  expense and your prompts come from real users (not generated synthetically
  from a narrow distribution).
- **Two-model setups** — you've already chosen a strong/weak pair you trust
  for your domain. Routing decides per-prompt which one runs.

## When **not** to reach for it

- **Single-model product** — if you've committed to one model end-to-end,
  routing has nothing to decide.
- **Prompts are very uniform** — e.g. you only do summarization. The router
  will mostly pick the same side and the overhead isn't worth it.
- **Latency budget is the constraint, not cost** — both models cost the
  same in latency? Pick the strong one and skip the router.

## What the demo does

`demo.py`:

1. Loads the pre-trained **matrix-factorization (mf)** router. It's small
   (a few MB), downloads to the HuggingFace cache, runs on CPU in
   milliseconds per prompt. (Pass `--router bert` to try the BERT one.)

2. Runs the router over a hand-curated list of prompts that span an obvious
   complexity range — from "what's the capital of France" through "explain
   why my React component re-renders" to "derive the Black-Scholes equation
   from first principles." Prints each prompt with its predicted
   `strong_win_rate`.

3. Sweeps over thresholds (0.1, 0.2, ..., 0.9) and reports, for each one,
   what fraction of prompts would be sent to the strong model. This is the
   knob you'd actually tune in production — pick the threshold that hits
   your target "% routed to strong" budget.

The demo deliberately **does not** call any LLM. We're showing what the
router decides, not running both models and comparing answers — that would
require API keys and budget, and the routing behavior is the interesting
part on its own.

## Running

```bash
uv sync --extra routellm
uv run python routellm/demo.py                    # default: mf router
uv run python routellm/demo.py --router bert      # try the BERT router
uv run python routellm/demo.py --threshold 0.3    # change the single-threshold cutoff
```

First run downloads ~50 MB of pre-trained router checkpoints from
HuggingFace (`routellm/mf_gpt4_augmented`, `routellm/bert_gpt4_augmented`).

## Reading the output

The `strong_win_rate` is the router's calibrated estimate that the strong
model would beat the weak model on this prompt. Higher = the strong model
is more clearly needed.

In production:

- **Static threshold mode**: pick one τ. Anything ≥ τ goes to strong.
- **Budget mode**: count what fraction you want to spend on the strong
  model, then pick the τ that hits that fraction on a sample of your
  traffic. The demo's sweep table is exactly the calibration plot you'd
  use to do this.

A common heuristic: pick τ so ~25–50% of traffic goes to strong. RouteLLM's
paper showed that captures most of the quality without much of the cost.
