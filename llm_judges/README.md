# LLM judges & routing — hands-on demos

Five self-contained subfolders, one per model. Each is a focused, well-documented
demo of a published LLM-quality or LLM-routing model: what it is, why it exists,
when you'd reach for it, and a runnable script that shows the key use case.

| Folder         | Model                       | What it does                                       | CPU OK? |
| ---            | ---                         | ---                                                | ---     |
| [prometheus/](prometheus/) | KAIST Prometheus 2          | Fine-grained LLM-as-judge with custom rubrics      | ❌ 7B+   |
| [pandalm/](pandalm/)       | WeOpenML PandaLM            | Pairwise judging with explanations                 | ❌ 7B    |
| [auto_j/](auto_j/)         | GAIR Auto-J (+ 4-bit)       | Critique-then-rate judge, single + pairwise        | ⚠️ 13B / 4-bit needs GPU |
| [hhem/](hhem/)             | Vectara HHEM-2.1-Open       | Hallucination/consistency scoring                  | ✅ ~110M |
| [routellm/](routellm/)     | LMSYS RouteLLM              | Routes prompts between a strong & weak LLM         | ✅ small |

## Why these five?

They cover the three places where "is this LLM output good?" or "how should I
use this LLM?" decisions actually happen in practice:

1. **General LLM-as-judge** (Prometheus, PandaLM, Auto-J) — open-weight
   alternatives to GPT-4-as-judge. You hand them a model output and they score
   or rank it. Useful for **offline eval of fine-tuning runs**, **RLHF / DPO
   preference data**, and **regression testing** of LLM apps.
2. **Hallucination detection** (HHEM) — a narrow but heavily-used task:
   given a source document and a generated summary, is the summary supported
   by the source? Cheap enough to run on every RAG response in production.
3. **Routing** (RouteLLM) — decide *before* generation whether a query needs
   a strong (expensive) or weak (cheap) model. Same answer quality at a
   fraction of the cost when the prompt mix is heterogeneous.

## Install

Each model has its own optional-dependencies extra so you can avoid pulling
multi-GB stacks you don't need:

```bash
cd llm_judges

uv sync --extra hhem            # just HHEM
uv sync --extra routellm        # just RouteLLM
uv sync --extra prometheus      # adds prometheus-eval (+ vLLM as a soft dep)
uv sync --extra pandalm
uv sync --extra auto_j
```

## Run a demo

```bash
uv run python hhem/demo.py
uv run python routellm/demo.py --router mf
uv run python prometheus/demo.py --model kaist-ai/prometheus-7b-v2.0
uv run python pandalm/demo.py
uv run python auto_j/demo.py --quantization 4bit
```

Every `demo.py` is single-file, top-to-bottom, with extensive comments
explaining *why* each step matters — not just what the line does. The
companion `README.md` in each folder is written for a data scientist
who wants to understand the model before running anything.

## Hardware notes

- **HHEM and RouteLLM** are small. A laptop CPU is fine.
- **Prometheus, PandaLM, Auto-J** are 7B–13B decoder LLMs. CPU inference
  technically works through HF transformers, but a single generation can
  take minutes. Realistic setups: a single 24 GB consumer GPU (Auto-J at
  4-bit fits), or A100/H100 for full precision.
- The 4-bit option in Auto-J depends on `bitsandbytes`, which only ships
  CUDA kernels. On Apple Silicon or CPU-only boxes, drop the `--quantization`
  flag and fall back to fp16 / bf16.

## Data

Every demo ships with **small inline examples** so you can sanity-check the
model in seconds. Where a standard benchmark is the natural choice (HaluEval
for HHEM, MT-Bench-style pairs for the judges, GSM8K-flavored prompts for
RouteLLM), the demo points to the HuggingFace dataset id so you can swap the
inline sample out for a real evaluation slice.
