---
license: apache-2.0
base_model: chaoliangUNSW/Jev-Style-2B-Decision-v4
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-classification
tags:
  - decision-model
  - decision-making
  - jev-style
  - system-one
  - calibration
  - long-context
  - qwen3.5
  - mlx
  - apple-silicon
  - on-device
  - llm-routing
  - guardrails
---

# Jev-Style-2B-Decision-v4-MLX

**Website:** [jevstyle.com](https://jevstyle.com) · **GitHub:** [jev-style](https://github.com/lawrence3699/jev-style)

**Jev-Style decision series:** [v1 · 2B](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16) → [v2 · 2B](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2-MLX-bf16) → [v3 · 0.8B](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX) → [v3 · 2B](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX) → **v4 · 2B**

**Jev-style decisions, native on Apple silicon.** These are the MLX builds of
[Jev-Style-2B-Decision-v4](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4) for text and JSON states:
bf16 (3.76 GB) and 8-bit (2.00 GB) in one repository, with one runtime (`jev_style_decision_mlx.py`; choose the
folder with `--model-dir bf16` or `--model-dir 8bit`). Images need the main repository's PyTorch runtime. Full
results, protocols, training data and licences are on the
[main model card](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4).

| | **Jev-Style 2B v4** | Jev-Style 2B v3 |
|---|---:|---:|
| Decision Index 0.3 public index (self-run, complete) | **44.37** | 34.05 |
| JevBench v1.4.1, public items (231) | **74.9%** | 73.6% |
| Context per call | **32,768 tokens** | 25,600 tokens |

<sub>Results of the model, measured with the main repository's PyTorch runtime: Decision Index public suite 0.3
with the kit's own runner and scorer (bfloat16 on one A100, one complete run of 140,178 requests; v3: its public
numbers on the 0.4.1 board); JevBench v1.4.1 public items with the official harness (float32 on Apple MPS, self-run
once, 95% CI 68.9–80.0%; v3: the number on its own card). Neither is an official board entry. These MLX builds were not re-run on the benchmarks;
they reproduce that runtime's scores within the pre-declared gates in the table below. Protocols:
[main card](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4#results).</sub>

## Precisions and parity

| Precision | Folder | Weights | Same top-1 as PyTorch FP32 (1,000 questions) | Max abs Δp | Accuracy (FP32: 84.4%) | Long fixture (97 questions) | Gate |
|---|---|---:|---:|---:|---:|---:|:---:|
| bf16 | `bf16/` | 3.76 GB | 99.7% | 0.036 | 84.4% | 97 / 97 | PASS |
| 8bit | `8bit/` | 2.00 GB | 99.9% | 0.048 | 84.5% | 97 / 97 | PASS |

- Each folder holds `model.safetensors`, its `config.json`, the tokenizer and `macjev_norms_fp32.safetensors`
  (FP32 norm weights, 0.42 MB, required). The top level holds the shared runtime `jev_style_decision_mlx.py`,
  `readout_config.json`, `release_config.json`, the sha256 `manifest.json`, `LICENSE`, `NOTICE`,
  `THIRD_PARTY_NOTICES.md` and a copy of `bf16/config.json` (for the Hub's download counter). Converted with mlx
  0.32.2 and mlx-lm 0.31.3.
- **Which folder:** `bf16/` for closeness to the FP32 reference (max abs Δp 0.036 vs 0.048, median
  0.0006 vs 0.0009); `8bit/` at about half the size when memory is tight: it made the same
  call as the FP32 reference on 99.9% of the 1,000 questions.

<sub>Reference: the main repository's reference scorer in float32 on CUDA (TF32 off, plain PyTorch Qwen3.5 layers),
exact block attention, on the released weights, text only; gates declared before any format was built or scored.
Gate fixture: 1,000 questions from our own held-out development rows (≤ 4,096 tokens; 645 choice, 316 true/false,
39 score; up to 151 options), accuracy = top-1 in the gold set. Long fixture: 70 requests / 97 questions, development
rows of 4,097 to 32,768 tokens plus edge cases (255 options, 15 catalogue questions, an input of exactly 32,768
tokens); the column counts questions with the same top-1 as the reference. On the long fixture the scores of every
question (bf16, 97 / 97) are closer to the block-attention reference than to plain-causal and prefix-causal
controls of the same scorer, so the runtime runs the attention the model was trained with, and in both folders `score_many` gave
exactly the scores of each question asked alone. MLX scored once on an Apple M1 Max. Sizes are the weight files
(GB = 10^9 bytes). Records: `validation/parity/`.</sub>

## Speed

On an Apple M1 Max (MLX bf16), a single question about an input of up to 1,024 tokens (median 250) took a median 233 ms, and one about 16,385–32,768 tokens (median 21,646) 15.52 s. `score_many` computes the state once for all questions about it.

| Input (state + question + options) | Requests | Median input | bf16 | 8bit |
|---|---:|---:|---:|---:|
| up to 1,024 tokens | 871 | 250 tokens | 233 ms | 249 ms |
| 1,025–4,096 tokens | 129 | 1,437 tokens | 1.01 s | 1.16 s |
| 4,097–16,384 tokens | 19 | 4,728 tokens | 3.29 s | 4.24 s |
| 16,385–32,768 tokens | 25 | 21,646 tokens | 15.52 s | 20.09 s |

- Pick a precision folder by size and closeness to FP32 (table above). bf16 and 8-bit were timed one after the other on the same machine; 8-bit was the slower folder in every row.

<sub>Apple M1 Max, 64 GB, macOS 15.7.5, mlx 0.32.2 / mlx-lm 0.31.3. Wall time of one scoring call (the input was rendered and the model loaded beforehand), the state computed from scratch, once per fixture request: the 1,000 gate-fixture requests and the single-question requests of the long fixture. The Mac was in normal use during the run (1-minute load average about 9), so treat the numbers as indicative.</sub>

## Quick start

```bash
pip install -U huggingface_hub
hf download chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX --local-dir Jev-Style-2B-Decision-v4-MLX
cd Jev-Style-2B-Decision-v4-MLX
pip install -r requirements.txt        # mlx 0.32.2, mlx-lm 0.31.3 (exactly), transformers 5.17.0, tokenizers, numpy
```

Only one precision is needed: add `--exclude "8bit/*"` to the download for bf16 only, or `--exclude "bf16/*"` for
8-bit only. Pass the precision folder to the runtime with `--model-dir bf16` or `--model-dir 8bit` (in Python,
`JevStyleDecisionMLX("bf16")` or `JevStyleDecisionMLX("8bit")`).

```bash
# choice question, bf16 weights
python jev_style_decision_mlx.py --model-dir bf16 --state "Finder is open on ~/Reports. The user asked: email report.pdf to Ana." \
  --question "What should the agent do next?" \
  --options '{"attach_report": "attach report.pdf to a new email to Ana", "rename_report": null, "close_finder": "close the Finder window"}'

# true/false question about a JSON state, 8-bit weights
python jev_style_decision_mlx.py --model-dir 8bit \
  --state-json '{"app": "Mail", "draft": {"to": "ana@example.com", "attachments": []}}' \
  --question "Is the report attached?" --qtype noul

# score (ordinal) question, uncalibrated probabilities (T = 1.0)
python jev_style_decision_mlx.py --model-dir 8bit --state "The deploy failed twice with the same migration error." \
  --question "How risky is retrying now?" --qtype score --options '["no risk", "some risk", "high risk"]' --temperature 1.0

# full typed question as JSON, first checking the sha256 of the runtime files and the chosen precision folder (manifest.json)
python jev_style_decision_mlx.py --model-dir bf16 --verify --state "cart: 3 items" \
  --question '{"t": "choice", "ins": "Next step?", "crit": {"checkout": null, "keep_shopping": null}}'
```

The output has `answer`, `probabilities` and raw `scores` per option, `temperature` (the calibrated temperature of
the question type: choice 0.770, true/false 0.825, score 0.584), `top_probability`, `entropy_concentration`,
`input_tokens`, `state_tokens`, `head_tokens`, `blocks` and `catalogue_overflow`.

```python
from jev_style_decision_mlx import JevStyleDecisionMLX
m = JevStyleDecisionMLX("8bit")                # the 8-bit precision folder
state = {"screen": "Settings > Wi-Fi", "goal": "join the network Office-5G"}
qs = [{"t": "choice", "ins": "Next action?", "crit": {"tap_office_5g": None, "toggle_wifi_off": None, "go_back": None}},
      {"t": "noul", "ins": "Is Wi-Fi turned on?", "crit": None}]
for r in m.score_many(state, qs):          # the state is computed once and reused for every question
    print(r["answer"], round(r["top_probability"], 4))
m.decide(state, qs[0])                     # same state again: reused (m.last_timing["state_reused"] is True)
m.close()
```

- Batch mode: `--jsonl requests.jsonl` (or `-` for stdin), rows `{"id"?, "state", "question", "options"?, "qtype"?,
  "temperature"?}`; consecutive rows with an identical state share one state computation, and a malformed row gets
  an `"error"` field while the batch continues.
- **Long inputs.** The complete input can be up to 32,768 tokens, with no separate question/options cap and up to
  255 options per question; question and options over 2,048 tokens use a numbered-option catalogue. In the parity run (table above) both folders scored an input of exactly
  32,768 tokens and a 255-option question (8,449 tokens, catalogue layout). A larger input raises `InputBudgetError` (CLI exit status 2); nothing is truncated.
- **mlx-lm must be exactly 0.31.3.** The runtime corrects two Qwen3.5 numerics inside mlx-lm and checks the source
  it patches; other versions are refused with the install command. It also refuses to run without
  `macjev_norms_fp32.safetensors` or `readout_config.json`, and never falls back to stock norms or T = 1.
- The input format, block attention and the readout are described on the
  [main card](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4#input-format-and-readout).

### With the jev-style package

[jev-style](https://github.com/lawrence3699/jev-style) 0.6.0 or later downloads this repository at a pinned revision
and runs this runtime behind a local systemone-compatible API, with a Playground, demos and MCP tools:

```bash
pip install "jev-style[mlx]"          # with mlx-lm 0.31.3
jev-style serve --release 2b-v4       # bf16 on Apple silicon; --precision 8bit for the 2.00 GB folder
```

```python
from jev_style import JevStyle, noul

js = JevStyle.from_pretrained("chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX", precision="8bit")
out = js.decide({"app": "Mail", "draft": {"to": "ana@example.com", "attachments": []}},
                {"attached": noul("Is the report attached?")})
```

## Scope and limits

- **Runtime required.** `mlx_lm.generate` can load the weights in `bf16/` or `8bit/` (the repository root holds no
  weights) but cannot produce the decision scores, and it would run causal attention. Use
  `jev_style_decision_mlx.py`.
- **Text only.** These files hold the language model; images in the state need the PyTorch runtime of the
  [main repository](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4).
- **32,768 tokens** is the limit for the whole input (state + question + options + readout); at most 255 options per
  question.
- **Decision Index.** The training data includes the official train splits of several benchmarks that the index
  tests on, so results on those benchmarks are not zero-shot; they are named on the
  [main card](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4#training-data-and-licences).
- **Decisions only.** The model scores the options you give it. It does not generate text and takes no actions.

## Disclaimers and licence

Apache-2.0. Built on Qwen/Qwen3.5-2B (Apache-2.0); `NOTICE` lists the modifications. **Soft labels distilled from
TypeSafe Jev 1.14 via its public API** are the targets of part of the training rows; no Jev weights or code are
included. Some training data has terms of its own and some training rows were written by Anthropic Claude and OpenAI
GPT models, whose providers' terms of use may restrict how models trained on them may be used: see [Training data
and licences](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4#training-data-and-licences) on the main
card. The runtime contains a modified copy of one mlx-lm 0.31.3 function (the Qwen3.5 `GatedDeltaNet.__call__`; MIT
License, Copyright Apple Inc.); its copyright and permission notice is in `THIRD_PARTY_NOTICES.md`. Not affiliated
with, endorsed by or connected to TypeSafe AI or Jev, the Laya authors or the Qwen team.

**AI disclosure:** code written with AI coding assistants (Claude Code) under my direction; I designed the project,
trained the models and verified the results.

## Contact

I welcome internship, employment, and research collaboration opportunities. Please contact me at [**yanchaoliang369@gmail.com**](mailto:yanchaoliang369@gmail.com).

欢迎提供实习、工作及科研合作机会，请邮件联系：[yanchaoliang369@gmail.com](mailto:yanchaoliang369@gmail.com)。
