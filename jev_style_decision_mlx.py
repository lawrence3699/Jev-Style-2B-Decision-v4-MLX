"""Jev-Style-2B-Decision-v4: typed decisions with MLX on Apple silicon (bf16 / 8-bit, --precision), text only.

Self-contained runtime for chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX (Apache-2.0). No dependency on any training
code: rendering (protocol "macjev-render-v2-long-options"), block-causal attention, the verdict readout and
calibration are implemented below and reproduce the reference implementation used for evaluation (see
release_config.json -> "runtime_parity"). Needs mlx, mlx-lm==0.31.3 (pinned, see the MLX backend section),
tokenizers and numpy.

Calibration: probabilities use the calibrated temperature of the question's type (readout_config.json ->
temperatures.by_qtype: choice 0.770, noul 0.825, score 0.584); temperature=... (CLI --temperature, JSONL
"temperature") overrides it (1.0 = uncalibrated scores). Unlike the 0.8B v3 runtime there is no --category (no group
temperatures) and no --head-max (no question/options cap: only the 32,768-token total budget applies).

Text only: images in the state need the main repository's PyTorch runtime.

Python:
    from jev_style_decision_mlx import JevStyleDecisionMLX
    m = JevStyleDecisionMLX(".", precision="bf16")          # or "8bit"
    m.decide("<state>", "Which action?", options={"click": "press it", "wait": None})
    m.score_many("<state>", [q1, q2, ...])                  # state computed once, reused for every question
"""
# ----------------------------------------------------------------------------------------------
# Shared core (byte-identical in jev_style_decision_gguf.py and jev_style_decision_mlx.py): input rendering (v2),
# verdict readout, calibration, budgets, errors, manifest check, CLI/JSONL. The main repository's PyTorch runtime calls
# the reference code instead; this renderer gives the same token ids, blocks and slots as that code for text
# (validation/runtime/render_check.json).
#
# Input layout ("macjev-render-v2-long-options", layout "sb"; token segments are encoded separately
# and concatenated, so slot positions are exact):
#
#   state prefix     State:\n<state>\n\n
#   short form       Question [<type>]: <question>\nOptions:\n
#   (question +      - <option 1> ->\n ... - <option K> ->\n
#    options + slots <= 2,048 tokens)
#   overflow form    Question [<type>]: <question>\nOptions:\n
#   (otherwise)      Option 1: <option 1>\n ... Option K: <option K>\n            (the catalogue)
#                    Judge each numbered option in the complete catalogue above:\n
#                    Option 1 ->\n ... Option K ->\n                               (the rubric)
#
# Attention blocks ([start, stop) token ranges): the state is cut into consecutive 2,048-token
# blocks; the short form is one block; in the overflow form the catalogue and the rubric are each cut
# into 2,048-token blocks. In the 6 full-attention layers every block attends to all earlier tokens
# and to itself with NO causal mask inside the block (block-causal); the Gated-DeltaNet layers are
# ordinary recurrent layers. A backend must compute exactly these blocks (never merged, never re-split).
#
# Score of option k = logit(" yes") - logit(" no") at its " ->" slot = h_slot . (W_yes - W_no) in
# float32 (final normed hidden state, tied embedding rows). Probabilities = softmax(scores / T) in
# canonical option order. T = the temperature of the question's type in readout_config.json
# (temperatures.by_qtype: choice / noul / score, fitted on 3,325 calibration rows) unless temperature=...
# overrides it (1.0 = uncalibrated scores). There are no category / group temperatures and no separate
# question/options budget, so the 0.8B v3 runtime's --category and --head-max options do not exist here.
#
# Budget: the complete input (state + question + options + readout) <= 32,768 tokens. Larger inputs
# raise InputBudgetError; nothing is ever truncated. Text inside the state, question and options is
# tokenised with special tokens disabled, so e.g. "<|im_end|>" in user text can never act as a
# control token. A non-finite score (NaN / inf) raises NonFiniteScoreError: no probabilities are made.
#
# Text only: this runtime does not read images ("<|image_1|>" in a state is plain text here). For images in the
# state use the main repository's PyTorch runtime (jev_style_decision.py).
# ----------------------------------------------------------------------------------------------
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = "Jev-Style-2B-Decision-v4"
TEMPLATE_VERSION = "macjev-render-v2-long-options"
READOUT_FORMAT = "macjev-readout-v2"
LAYOUT = "sb"
BLOCK = 2048                    # attention block size (a processing unit, not a content limit)
CONTEXT_LIMIT = 32_768          # state + question + options + readout, all included
QTYPES = ("choice", "score", "noul")
JSONL_GROUP_MAX = 256           # consecutive JSONL rows with one state scored in one backend call
HERE = Path(__file__).resolve().parent


class InputBudgetError(ValueError):
    """The rendered input exceeds the token budget. Nothing was truncated."""


class QuestionError(ValueError):
    """The question/options are malformed."""


class NonFiniteScoreError(FloatingPointError):
    """The model produced a non-finite decision score (NaN or inf). No probabilities are returned."""


# -- questions ------------------------------------------------------------------------------------
def option_names(question):
    """Canonical option identifiers, in the order the probabilities are returned."""
    if not isinstance(question, dict):
        raise QuestionError("question must be a dict {'t', 'ins', 'crit'}")
    t, crit = question.get("t"), question.get("crit")
    if not isinstance(question.get("ins"), str) or not question["ins"].strip():
        raise QuestionError("question text ('ins') must be a non-empty string")
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise QuestionError("choice needs a non-empty dict {option name: description or None}")
        if len(crit) > 255:
            raise QuestionError(f"choice supports at most 255 options (got {len(crit)})")
        return [str(k) for k in crit]
    if t == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise QuestionError("score needs a list of 2..10 level descriptions")
        return [str(i) for i in range(len(crit))]
    if t == "noul":
        if crit is not None and not isinstance(crit, dict):
            raise QuestionError("noul criteria must be None or {'false': ..., 'true': ...}")
        return ["false", "true"]
    raise QuestionError(f"unknown question type {t!r} (expected one of {QTYPES})")


def make_question(question, options=None, qtype=None):
    """Build a typed question.

    * ``question`` already a dict {"t", "ins", "crit"}: validated and returned.
    * ``qtype="choice"`` (default when ``options`` is given): ``options`` = {name: description or None}
      or a list of names.
    * ``qtype="score"``: ``options`` = list of 2..10 level descriptions (level 0 first).
    * ``qtype="noul"`` (default when no options): a true/false statement; ``options`` may be
      {"false": "...", "true": "..."} to describe the two outcomes.
    """
    if isinstance(question, dict):
        q = dict(question)
    else:
        if qtype is None:
            qtype = "choice" if options is not None else "noul"
        if qtype == "choice":
            if isinstance(options, (list, tuple)):
                if len(set(map(str, options))) != len(options):
                    raise QuestionError("duplicate option names")
                crit = {str(o): None for o in options}
            else:
                crit = options
        elif qtype == "score":
            crit = list(options) if options is not None else None
        else:
            crit = options
        q = {"t": qtype, "ins": question, "crit": crit}
    option_names(q)
    return q


def serialize_state(state):
    """Strings pass through unchanged; any other JSON value is serialised (ensure_ascii=False)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def _criterion(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question):
    t, crit = question["t"], question.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else f"{k}: {_criterion(v)}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {_criterion(c)}" for i, c in enumerate(crit)]
    crit = crit or {}
    false_c, true_c = crit.get("false"), crit.get("true")
    return ["false: " + (_criterion(false_c) if false_c not in (None, "") else "no, the statement does not hold"),
            "true: " + (_criterion(true_c) if true_c not in (None, "") else "yes, the statement holds")]


# -- tokenizer + renderer -----------------------------------------------------------------------
class TextEncoder:
    """HF ``tokenizers`` tokenizer.json; no BOS/EOS, special tokens in text are split (never control tokens)."""

    def __init__(self, tokenizer_json):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(tokenizer_json))
        self.tk.encode_special_tokens = True

    def __call__(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def id_to_token(self, i):
        return self.tk.id_to_token(int(i))

    def vocab_size(self):
        return self.tk.get_vocab_size(with_added_tokens=True)


def _blocks(start, stop):
    return [(s, min(s + BLOCK, stop)) for s in range(start, stop, BLOCK)]


class Rendered:
    """One rendered question: token ids, the verdict slots (one per option, canonical order), the
    attention blocks ([start, stop) pairs tiling [0, len(ids))) and the state prefix length."""
    __slots__ = ("ids", "prefix_len", "slots", "blocks", "names", "qtype", "catalogue_overflow")

    def __init__(self, ids, prefix_len, slots, blocks, names, qtype, catalogue_overflow):
        self.ids, self.prefix_len, self.slots, self.blocks = ids, prefix_len, slots, blocks
        self.names, self.qtype, self.catalogue_overflow = names, qtype, catalogue_overflow

    @property
    def state_blocks(self):
        return [b for b in self.blocks if b[1] <= self.prefix_len]

    @property
    def question_blocks(self):
        return [b for b in self.blocks if b[0] >= self.prefix_len]

    @property
    def input_tokens(self):
        return len(self.ids)

    @property
    def head_tokens(self):
        """question + options + readout tokens (everything after the state prefix)"""
        return len(self.ids) - self.prefix_len


class Renderer:
    def __init__(self, encode, readout_cfg, max_len=CONTEXT_LIMIT):
        want = {"format": READOUT_FORMAT, "template": TEMPLATE_VERSION, "layout": LAYOUT, "readout": "verdict",
                "block_size": BLOCK, "total_context_limit": CONTEXT_LIMIT}
        bad = {k: readout_cfg.get(k) for k, v in want.items() if readout_cfg.get(k) != v}
        if bad:
            raise ValueError(f"readout_config.json does not describe this runtime's protocol {want}; got {bad}")
        if not 0 < int(max_len) <= CONTEXT_LIMIT:
            raise ValueError(f"max_len must be in 1..{CONTEXT_LIMIT}")
        self.enc, self.max_len = encode, int(max_len)
        self.head_max = self.max_len    # no separate question/options cap (field kept for the 0.8B / jev-style API)
        st = readout_cfg["slot_tokens"]
        self.yes, self.no, arrow = int(st["yes"]["id"]), int(st["no"]["id"]), int(st["verdict_slot"]["id"])
        for text, want_id in ((" yes", self.yes), (" no", self.no), (" ->", arrow)):
            got = self.enc(text)
            if got != [want_id]:
                raise ValueError(f"tokenizer mismatch: {text!r} -> {got}, readout_config expects [{want_id}]")
        self.arrow = [arrow]
        self.newline = self.enc("\n")
        self.dash = self.enc("- ")
        self.rubric_head = self.enc("Judge each numbered option in the complete catalogue above:\n")
        self._numbered = {}

    def _option_label(self, pos, catalogue):
        key = (pos, catalogue)
        if key not in self._numbered:
            self._numbered[key] = self.enc(f"Option {pos + 1}: " if catalogue else f"Option {pos + 1}")
        return self._numbered[key]

    def prefix_ids(self, state):
        return self.enc("State:\n") + self.enc(serialize_state(state)) + self.enc("\n\n")

    def pieces(self, state, question, prefix=None):
        """Tokenised segments of one question (``prefix``: already tokenised state, to tokenise it once)."""
        names = option_names(question)
        return {"prefix": self.prefix_ids(state) if prefix is None else list(prefix),
                "head": self.enc(f"Question [{question['t']}]: {question['ins']}\nOptions:\n"),
                "opts": [self.enc(o) for o in render_options(question)], "names": names, "qtype": question["t"]}

    def assemble(self, pieces, max_len=None):
        maximum = self.max_len if max_len is None else min(self.max_len, int(max_len))
        prefix, head, opts = list(pieces["prefix"]), list(pieces["head"]), pieces["opts"]
        k = len(opts)
        if k < 1:
            raise QuestionError("at least one option is required")
        short, rel = list(head), []
        for o in opts:
            short += self.dash + list(o) + self.arrow
            rel.append(len(short) - 1)
            short += self.newline
        overflow = len(short) > BLOCK
        if not overflow:
            ids = prefix + short
            slots = [len(prefix) + s for s in rel]
            blocks = _blocks(0, len(prefix)) + [(len(prefix), len(ids))]
        else:
            catalogue = list(head)
            for pos, o in enumerate(opts):
                catalogue += self._option_label(pos, True) + list(o) + self.newline
            doc_end = len(prefix) + len(catalogue)
            rubric, rel = list(self.rubric_head), []
            for pos in range(k):
                rubric += self._option_label(pos, False) + self.arrow
                rel.append(len(rubric) - 1)
                rubric += self.newline
            ids = prefix + catalogue + rubric
            slots = [doc_end + s for s in rel]
            blocks = _blocks(0, len(prefix)) + _blocks(len(prefix), doc_end) + _blocks(doc_end, len(ids))
        if len(ids) > maximum:
            raise InputBudgetError(f"the complete input needs {len(ids)} tokens (state {len(prefix)} + question/"
                                   f"options/readout {len(ids) - len(prefix)}); the limit is {maximum} tokens (model "
                                   f"maximum {CONTEXT_LIMIT}). Nothing was truncated: shorten the state, the "
                                   f"question or the options.")
        return Rendered(ids, len(prefix), slots, blocks, list(pieces["names"]), pieces["qtype"], overflow)

    def render(self, state, question, max_len=None, prefix=None):
        return self.assemble(self.pieces(state, question, prefix), max_len)


# -- calibration ----------------------------------------------------------------------------------
def check_temperature(t):
    try:
        t = float(t)
    except (TypeError, ValueError):
        raise ValueError(f"temperature must be a number, got {t!r}") from None
    if not (math.isfinite(t) and t > 0):
        raise ValueError(f"temperature must be finite and > 0, got {t!r}")
    return t


def softmax_probabilities(scores, temperature):
    """softmax(scores / T) in float64 (the reference computation)."""
    z = np.asarray(scores, float) / float(temperature)
    z = np.exp(z - z.max())
    return z / z.sum()


def concentration(p):
    k = len(p)
    if k < 2:
        return 1.0
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


NOT_VERIFIED = ("README.md", "eval_results.json")          # documentation / records: in manifest.json, not checked
NOT_VERIFIED_DIRS = ("assets/", "figures/", "validation/")


def verify_manifest(model_dir, only=None):
    """Re-hash the files listed in manifest.json (all, or those whose path starts with one of ``only``).
    Documentation and evaluation records (README.md, eval_results.json, assets/, figures/, validation/) are
    recorded in the manifest but not checked here (not even when named in ``only``), so a card edit never
    makes the runtime refuse to load and a download without them still verifies."""
    model_dir = Path(model_dir)
    man = json.loads((model_dir / "manifest.json").read_text())
    bad, missing, checked = [], [], 0
    for name, rec in man["files"].items():
        if name in NOT_VERIFIED or name.startswith(NOT_VERIFIED_DIRS):
            continue
        if only and not any(name == o or name.startswith(o.rstrip("/") + "/") for o in only):
            continue
        p = model_dir / name
        if not p.exists():
            missing.append(name)
        elif _sha256(p) != rec["sha256"]:
            bad.append(name)
        checked += 1
    return {"ok": not bad and not missing, "checked": checked, "bad": bad, "missing": missing}


class DecisionBase:
    """Backend-independent part. A backend implements ``_scores_many(rendered) -> list of list[float]``:
    ``rendered`` is a non-empty list of Rendered that all share ONE state (identical prefix ids and
    state blocks); it returns the raw float32 scores of every slot of every Rendered (slot order =
    canonical option order), computing the state blocks once per call. Backends keep the most recent
    state for the next call (exact prefix match only), so consecutive calls about the same state (e.g.
    JSONL rows read from stdin) reuse it."""
    backend = "base"

    def _setup(self, model_dir, tokenizer_json, max_len=CONTEXT_LIMIT, temperature=None):
        self.model_dir = Path(model_dir)
        self.readout_config = json.loads((self.model_dir / "readout_config.json").read_text())
        by_qtype = self.readout_config["temperatures"]["by_qtype"]
        missing = [q for q in QTYPES if q not in by_qtype]
        if missing:
            raise ValueError(f"readout_config.json has no calibrated temperature for {missing}")
        self.calibrated_temperatures = {q: check_temperature(by_qtype[q]) for q in QTYPES}
        self.default_temperature = None if temperature is None else check_temperature(temperature)
        self.encode = TextEncoder(tokenizer_json)
        self.renderer = Renderer(self.encode, self.readout_config, max_len=max_len)

    def temperature_for(self, qtype):
        """The temperature used for questions of this type (the override given to the constructor, if any)."""
        if self.default_temperature is not None:
            return self.default_temperature
        return self.calibrated_temperatures[qtype]

    def _scores_many(self, rendered):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _score_all(self, rendered):
        """Raw scores (lists of floats, canonical order) for Rendered sharing one state; one backend call."""
        if not rendered:
            return []
        p = rendered[0].prefix_len
        prefix, sblocks = rendered[0].ids[:p], rendered[0].state_blocks
        for r in rendered[1:]:
            if r.prefix_len != p or r.ids[:p] != prefix or r.state_blocks != sblocks:
                raise ValueError("all questions of one backend call must share the same state")
        scores = self._scores_many(rendered)
        if len(scores) != len(rendered) or any(len(s) != len(r.slots) for s, r in zip(scores, rendered)):
            raise RuntimeError("backend returned a wrong number of scores")
        return [[float(x) for x in s] for s in scores]

    def score_pieces(self, pieces_list, max_len=None):
        """Raw scores for pre-tokenised questions sharing one state: dicts {"prefix", "head", "opts",
        "names", "qtype"} of token ids (see Renderer.pieces). For parity checks; no temperature."""
        return self._score_all([self.renderer.assemble(p, max_len) for p in pieces_list])

    def _result(self, r, scores, temperature):
        t = self.temperature_for(r.qtype) if temperature is None else check_temperature(temperature)
        bad = [n for n, x in zip(r.names, scores) if not math.isfinite(x)]
        if bad:
            raise NonFiniteScoreError(f"non-finite decision scores for option(s) {bad} ({len(r.ids)} input tokens); "
                                      f"refusing to return probabilities. Check the weights file / engine build.")
        p = softmax_probabilities(scores, t)
        if not np.all(np.isfinite(p)):
            raise NonFiniteScoreError("non-finite probabilities")
        i = int(p.argmax())
        return {"answer": r.names[i], "probabilities": dict(zip(r.names, p.tolist())),
                "scores": dict(zip(r.names, scores)), "temperature": t, "top_probability": float(p[i]),
                "entropy_concentration": concentration(p), "input_tokens": len(r.ids),
                "state_tokens": r.prefix_len, "head_tokens": r.head_tokens, "image_tokens": None,
                "blocks": len(r.blocks), "catalogue_overflow": r.catalogue_overflow, "model": MODEL_NAME,
                "backend": self.backend}

    def decide(self, state, question, options=None, qtype=None, category=None, temperature=None, head_max=None,
               max_len=None):
        """Score one question about ``state``.

        Returns {"answer", "probabilities" {option: p}, "scores" {option: logit(yes)-logit(no)},
        "temperature", "top_probability", "entropy_concentration", "input_tokens", "state_tokens",
        "head_tokens", "image_tokens" (always None: text only), "blocks", "catalogue_overflow", "model",
        "backend"}. ``temperature`` overrides the calibrated temperature of the question's type (1.0 =
        uncalibrated scores). ``category`` and ``head_max`` are accepted for compatibility with the 0.8B v3
        runtime and the jev-style package, and ignored: this model has one temperature per question type and
        no separate question/options budget. Raises InputBudgetError (never truncates), QuestionError or
        NonFiniteScoreError."""
        q = make_question(question, options, qtype)
        r = self.renderer.render(state, q, max_len)
        return self._result(r, self._score_all([r])[0], temperature)

    def score_many(self, state, questions, category=None, temperature=None, head_max=None, max_len=None):
        """Several questions about ONE state: the state is tokenised and computed once and reused for
        every question (one backend call). ``questions``: dicts {"t","ins","crit"} (or anything
        make_question accepts). Results in order, identical to calling decide() per question. All
        questions are rendered and budget-checked before any scoring. ``category`` / ``head_max``: see
        decide() (accepted and ignored)."""
        qs = [make_question(q) for q in questions]
        if not qs:
            return []
        prefix = self.renderer.prefix_ids(state)
        rs = [self.renderer.render(state, q, max_len, prefix=prefix) for q in qs]
        return [self._result(r, sc, temperature) for r, sc in zip(rs, self._score_all(rs))]

    decide_many = score_many


def base_arg_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model-dir", default=str(HERE), help="folder with the weights and readout_config.json")
    ap.add_argument("--state", help="state as plain text")
    ap.add_argument("--state-json", help="state as a JSON value")
    ap.add_argument("--question", help="question text (or a JSON question {'t','ins','crit'})")
    ap.add_argument("--options", help="JSON: {name: description} or [names] (choice); [levels] (score)")
    ap.add_argument("--qtype", choices=QTYPES)
    ap.add_argument("--temperature", type=float,
                    help="override the calibrated per-type temperatures of readout_config.json (1.0 = raw scores)")
    ap.add_argument("--max-len", type=int, default=CONTEXT_LIMIT,
                    help=f"total token budget (state + question + options + readout), at most {CONTEXT_LIMIT}")
    ap.add_argument("--jsonl", help="batch mode: input JSON lines {id?, state, question, options?, qtype?, "
                                    "temperature?} ('-' = stdin); one JSON result per line on stdout. Consecutive "
                                    "rows with an identical state share one state computation")
    ap.add_argument("--verify", action="store_true", help="check sha256 of the files in manifest.json first")
    return ap


def _jsonl_groups(src, streaming):
    """(line number, record or error text) grouped into runs of consecutive rows with one state. From a
    file up to JSONL_GROUP_MAX rows are grouped; from stdin every row is its own group (answered at once;
    the backend's kept state still makes consecutive identical states cheap)."""
    group, key = [], None
    for n, line in enumerate(src):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError("a JSONL row must be a JSON object")
            k = serialize_state(rec.get("state", ""))
        except ValueError as e:
            if group:
                yield group
            group, key = [], None
            yield [(n, f"{type(e).__name__}: {e}")]
            continue
        if group and (k != key or len(group) >= JSONL_GROUP_MAX):
            yield group
            group = []
        group.append((n, rec))
        key = k
        if streaming:
            yield group
            group, key = [], None
    if group:
        yield group


def _run_group(engine, group, args):
    """Score one group of JSONL rows sharing a state. Returns (output rows, non-finite count)."""
    out, todo = {}, []
    prefix = None
    for n, rec in group:
        if isinstance(rec, str):
            out[n] = {"id": n, "error": rec}
            continue
        rid = rec.get("id", n)
        try:
            if "question" not in rec:
                raise QuestionError("row has no 'question'")
            q = make_question(rec["question"], options=rec.get("options"), qtype=rec.get("qtype"))
            t = rec.get("temperature", args.temperature)
            t = None if t is None else check_temperature(t)
            if prefix is None:
                prefix = engine.renderer.prefix_ids(rec.get("state", ""))
            todo.append((n, rid, engine.renderer.render(rec.get("state", ""), q, prefix=prefix), t))
        except (InputBudgetError, QuestionError, ValueError, TypeError, AttributeError) as e:
            out[n] = {"id": rid, "error": f"{type(e).__name__}: {e}"}
    nonfinite = 0
    if todo:
        scores = engine._score_all([r for _, _, r, _ in todo])
        for (n, rid, r, t), sc in zip(todo, scores):
            try:
                out[n] = {"id": rid, **engine._result(r, sc, t)}
            except NonFiniteScoreError as e:
                nonfinite += 1
                print(f"ERROR row {rid}: NonFiniteScoreError: {e}", file=sys.stderr, flush=True)
                out[n] = {"id": rid, "error": f"NonFiniteScoreError: {e}"}
    return [out[n] for n, _ in group], nonfinite


def run_cli(args, engine):
    """Exit status: 0 = ok (JSONL rows with input errors carry an "error" field), 2 = input error
    (single question), 3 = at least one non-finite score (refused, see stderr)."""
    if args.jsonl:
        streaming = args.jsonl == "-"
        src = sys.stdin if streaming else open(args.jsonl, encoding="utf-8")
        nonfinite = 0
        try:
            for group in _jsonl_groups(src, streaming):
                rows, bad = _run_group(engine, group, args)
                nonfinite += bad
                for row in rows:
                    print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            if not streaming:
                src.close()
        return 3 if nonfinite else 0
    if args.question is None:
        raise SystemExit("--question (or --jsonl) is required")
    try:
        state = json.loads(args.state_json) if args.state_json is not None else (args.state or "")
        question = args.question
        if question.lstrip().startswith("{"):
            try:                                            # a JSON question {'t','ins','crit'}; else plain text
                parsed = json.loads(question)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                question = parsed
        options = json.loads(args.options) if args.options else None
        res = engine.decide(state, question, options=options, qtype=args.qtype, temperature=args.temperature)
    except (InputBudgetError, QuestionError, ValueError, TypeError) as e:  # NonFiniteScoreError is not a ValueError
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except NonFiniteScoreError as e:
        print(f"ERROR: NonFiniteScoreError: {e}", file=sys.stderr)
        return 3
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0
# ---------------------------------------------------------------------------- end of shared core


# ---------------------------------------------------------------------------------- MLX backend
# Repository layout (one runtime, two precisions):
#
#   jev_style_decision_mlx.py, readout_config.json, release_config.json, manifest.json, ...   shared
#   bf16/   config.json, model.safetensors (bfloat16), macjev_norms_fp32.safetensors, tokenizer ...
#   8bit/   config.json, model.safetensors (affine 8-bit, group 64), macjev_norms_fp32.safetensors, tokenizer ...
#
# Either folder may be downloaded alone (plus the shared top-level files).
#
# Block-causal attention on mlx-lm: the input is fed to the model ONE protocol block per call on an mlx-lm
# prompt cache. The 6 full-attention layers use BlockKVCache, whose make_mask returns None, so the queries of
# the current block attend to every cached key plus every key of the block itself, with no causal mask inside
# the block. The 18 Gated-DeltaNet layers keep their recurrent cache (conv + recurrent state), causal by
# construction. The state blocks are computed once; each question then runs its block(s) on an independent
# copy of that cache, so questions never see each other and the state is reused (score_many, JSONL rows).
#
# Two corrections to stock mlx-lm 0.31.3 (both needed to match the HF / training numerics):
#   1. Gated-DeltaNet q/k normalisation: mlx-lm uses rms_norm(x, eps=1e-6) (= l2norm with eps Dk*1e-6); HF / fla /
#      llama.cpp use l2norm(x, eps=1e-6). A copy of mlx-lm's GatedDeltaNet.__call__ with only that eps changed is
#      swapped in per instance (never globally).
#   2. Qwen3.5 RMSNorm is x_hat * (1 + w). mlx-lm's converter adds the 1 in bf16, so the export stores bf16(1 + w).
#      The exact FP32 (1 + w) ships as the sidecar macjev_norms_fp32.safetensors (ignored by stock mlx-lm); the
#      runtime installs it and computes those norms in FP32 (cast back to the input dtype, as HF does).
# Both depend on mlx-lm internals, so the runtime checks the sha256 of the mlx-lm source it patches or relies on
# and refuses to run on any other mlx-lm version (install mlx-lm==0.31.3). The sidecar and readout_config.json are
# mandatory: the runtime never falls back to stock norms or to T = 1.
REPO_ID = "chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX"
PRECISIONS = ("bf16", "8bit")
DEFAULT_PRECISION = "bf16"
# activation dtype per precision: None = the checkpoint's native activations (bf16 residual stream), the
# setting the release scoring used for both precisions (release_config.json -> runtime.precisions overrides)
VALIDATED_COMPUTE_DTYPE = {"bf16": None, "8bit": None}
MLX_CACHE_LIMIT_GIB = 2.0
MLX_LM_VERSION = "0.31.3"
NORM_SIDECAR = "macjev_norms_fp32.safetensors"
NORM_SIDECAR_FORMAT = "macjev-norms-fp32-v1"
SHIFTED_NORMS = ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm")
# sha256 of inspect.getsource(...) in mlx-lm 0.31.3 for every function the MLX path patches (GatedDeltaNet) or relies
# on for the block mask (a None mask from BlockKVCache must reach scaled_dot_product_attention unchanged)
MLX_LM_SOURCE_SHA256 = {
    "qwen3_5.GatedDeltaNet.__call__": "9f27bdc613912fa9b3b0509d8e3f684ac672810bf45e29831f5815a1c3b43091",
    "qwen3_5.Qwen3_5TextModel.__call__": "45fd379f8885e93b4ba685c3d6c2aa1b891d535bb00fd042c6b1e56613345b6d",
    "base.create_attention_mask": "a8ebacf963b3f96e9ad838a7a870d1ef6da33aec5233b133b0d0c744c3b33e5e",
    "qwen3_5.Attention.__call__": "a938424855f1b919d1bcfd62b105e10723f7de295f5ee51a19924b502ddc86a6",
    "base.scaled_dot_product_attention": "8deba4a8a8fb4e29f1e69ea81701da68f08f90656e1421a1e7a82332aadc41c1",
}


class MLXVersionError(RuntimeError):
    """The installed mlx-lm is not the version whose internals the runtime was validated against."""


def check_mlx_lm():
    """Raise MLXVersionError unless the mlx-lm source this runtime patches is byte-identical to mlx-lm 0.31.3."""
    import inspect
    import mlx_lm
    from mlx_lm.models import base, qwen3_5
    objs = {"qwen3_5.GatedDeltaNet.__call__": qwen3_5.GatedDeltaNet.__call__,
            "qwen3_5.Qwen3_5TextModel.__call__": qwen3_5.Qwen3_5TextModel.__call__,
            "base.create_attention_mask": base.create_attention_mask,
            "qwen3_5.Attention.__call__": qwen3_5.Attention.__call__,
            "base.scaled_dot_product_attention": base.scaled_dot_product_attention}
    changed = [k for k, f in objs.items()
               if hashlib.sha256(inspect.getsource(f).encode()).hexdigest() != MLX_LM_SOURCE_SHA256[k]]
    if changed:
        raise MLXVersionError(
            f"this runtime needs mlx-lm=={MLX_LM_VERSION} (installed: {getattr(mlx_lm, '__version__', '?')}): it "
            f"corrects the Qwen3.5 Gated-DeltaNet q/k norm and the (1 + w) RMSNorm weights inside mlx-lm, and the "
            f"source it patches differs ({', '.join(changed)}). Install the tested version with: "
            f"pip install 'mlx-lm=={MLX_LM_VERSION}'")


# Third-party code: _gdn_hf_call below is a modified copy of GatedDeltaNet.__call__ from mlx_lm/models/qwen3_5.py,
# mlx-lm 0.31.3 (https://github.com/ml-explore/mlx-lm), MIT License (mlx-lm LICENSE: "Copyright © 2023 Apple Inc.";
# qwen3_5.py header: "Copyright © 2026 Apple Inc."). Changes:
# q/k rms_norm eps 1e-6 -> 1e-6 / Dk (HF-exact l2norm), sharding branches removed (refused), module-level function.
# The MIT copyright and permission notice is reproduced in THIRD_PARTY_NOTICES.md of this repository.
def _gdn_hf_call(self, inputs, mask=None, cache=None):
    """mlx-lm 0.31.3 qwen3_5.GatedDeltaNet.__call__ with HF-exact q/k l2norm (eps 1e-6). No sharding."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import gated_delta_update
    if self.sharding_group is not None:
        raise ValueError("sharded Gated-DeltaNet is not supported by this runtime")
    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = self.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(self.conv1d(conv_input))
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
        mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
        [self.num_k_heads, self.num_k_heads, self.num_v_heads],
        [self.head_k_dim, self.head_k_dim, self.head_v_dim])]
    state = cache[1] if cache else None
    dk = k.shape[-1]
    inv_scale = dk ** -0.5
    # l2norm(x, 1e-6) == rms_norm(x, eps=1e-6/Dk) / sqrt(Dk); q additionally carries the 1/sqrt(Dk) scale
    q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6 / dk)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6 / dk)
    out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
                                    use_kernel=not self.training)
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


def patch_gdn_l2norm(inner):
    """Swap every GatedDeltaNet instance of ``inner`` to the HF-exact l2norm class (per instance). -> count."""
    from mlx_lm.models.qwen3_5 import GatedDeltaNet
    cls = type("GatedDeltaNetHFL2", (GatedDeltaNet,), {"__call__": _gdn_hf_call})
    n = 0
    for layer in inner.layers:
        if getattr(layer, "is_linear", False):
            if type(layer.linear_attn) is not GatedDeltaNet:
                raise TypeError(f"unexpected Gated-DeltaNet class {type(layer.linear_attn).__name__}")
            layer.linear_attn.__class__ = cls
            n += 1
    return n


def load_norm_sidecar(path):
    """{mlx parameter name: FP32 (1 + w)} from macjev_norms_fp32.safetensors (format checked)."""
    import mlx.core as mx
    arrays, meta = mx.load(str(path), return_metadata=True)
    if (meta or {}).get("format") != NORM_SIDECAR_FORMAT:
        raise ValueError(f"{path} is not a {NORM_SIDECAR_FORMAT} norm sidecar (metadata {meta!r})")
    mx.eval(arrays)                                   # read now (mx.load is lazy)
    return arrays


def apply_exact_norms(inner, norms):
    """Install FP32 (1 + w) into every shifted RMSNorm of ``inner`` (per-instance class swap). -> count.
    Refuses a sidecar that misses a norm, names an unknown module or does not belong to these weights."""
    import mlx.core as mx
    import mlx.nn as nn

    def _call(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(x.dtype)

    cls = type("RMSNormFP32Weight", (nn.RMSNorm,), {"__call__": _call})
    modules = dict(inner.named_modules())
    seen = set()
    for name, w in norms.items():
        mod_name = name[:-len(".weight")] if name.endswith(".weight") else name
        mod = modules.get(mod_name)
        if mod is None or type(mod) is not nn.RMSNorm:
            raise KeyError(f"norm sidecar entry {name} is not an RMSNorm of this model")
        w = w.astype(mx.float32)
        if w.shape != mod.weight.shape:
            raise ValueError(f"norm sidecar shape mismatch at {name}")
        # the sidecar must be the (1 + w) of THESE weights (up to the bf16 rounding it corrects)
        if float(mx.abs(w - mod.weight.astype(mx.float32)).max()) > 2 ** -7 * float(mx.abs(w).max()) + 1e-6:
            raise ValueError(f"norm sidecar does not match the model weights at {name}")
        mod.weight = w
        mod.__class__ = cls
        seen.add(mod_name)
    missing = [k for k, m in modules.items() if type(m) is nn.RMSNorm and k not in seen
               and (k == "norm" or k.endswith(SHIFTED_NORMS))]
    if missing:
        raise KeyError(f"norm sidecar misses {missing[:3]}")
    mx.eval(inner.parameters())
    return len(seen)


def _block_kv_class():
    from mlx_lm.models.cache import KVCache

    class BlockKVCache(KVCache):
        """KV cache whose make_mask is None: the current block attends to all cached keys and to itself
        (Qwen3_5TextModel builds one mask from the first full-attention layer's cache)."""

        def make_mask(self, *args, **kwargs):
            return None

    return BlockKVCache


def copy_cache(cache):
    """Independent copy of a prompt cache (KV + ArraysCache). KV copies are trimmed to ``offset`` (the next
    update re-allocates by concatenation); ArraysCache gets a new list (GDN layers assign new arrays)."""
    import copy
    from mlx_lm.models.cache import ArraysCache, KVCache
    out = []
    for c in cache:
        n = copy.copy(c)
        if isinstance(c, KVCache):
            if c.keys is not None:
                n.keys = c.keys[..., :c.offset, :]
                n.values = c.values[..., :c.offset, :]
        elif isinstance(c, ArraysCache):
            n.cache = list(c.cache)
            if c.lengths is not None or c.left_padding is not None:
                raise ValueError("batched ArraysCache state is not supported")
        else:
            raise TypeError(f"unsupported cache type {type(c).__name__}")
        out.append(n)
    return out


def _set_mlx_cache_limit(mx, gib):
    """Bound MLX's freed-buffer cache (process-wide); never raises a lower limit set by the host."""
    if gib is None:
        return None
    want = int(float(gib) * (1 << 30))
    prev = mx.set_cache_limit(want)
    if prev is not None and prev < want:
        mx.set_cache_limit(prev)
        return int(prev)
    return want


def _weights_precision(weights_dir):
    """'bf16' or '8bit' (or '<n>bit') from the MLX config.json of a weights folder."""
    cfg = json.loads((Path(weights_dir) / "config.json").read_text())
    q = cfg.get("quantization") or cfg.get("quantization_config")
    return f"{int(q['bits'])}bit" if q else "bf16"


def _require_weights(w, precision, repo):
    need = ["config.json", "model.safetensors", "tokenizer.json", NORM_SIDECAR]
    miss = [n for n in need if not (w / n).is_file()]
    if miss:
        extra = (f" {NORM_SIDECAR} holds the exact FP32 (1 + w) RMSNorm weights; without it mlx-lm would use "
                 f"bf16-rounded norms, so the runtime refuses to load." if NORM_SIDECAR in miss else "")
        raise FileNotFoundError(f"{w} is missing {', '.join(miss)}.{extra} Download the folder with: "
                                f"hf download {REPO_ID} --include \"{precision}/*\" --local-dir {repo}")


def resolve_model_dirs(model_dir=HERE, precision=None):
    """-> (repo_dir, weights_dir, precision).

    ``model_dir`` is the repository folder (holding bf16/ and/or 8bit/ next to readout_config.json),
    or one precision folder itself (then ``precision`` defaults to the precision stored there).
    ``precision`` None means "bf16" for a repository folder."""
    if precision is not None and precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    d = Path(model_dir).resolve()
    # a precision folder given directly (the repository root also holds a config.json, a copy of bf16/config.json,
    # so a folder counts as a precision folder only if it has its own weights or no bf16/ / 8bit/ subfolder)
    if (d / "config.json").is_file() and ((d / "model.safetensors").is_file()
                                          or not any((d / p).is_dir() for p in PRECISIONS)):
        found = _weights_precision(d)
        if precision is not None and precision != found:
            raise ValueError(f"{d} holds {found} weights, not {precision}")
        repo = d if (d / "readout_config.json").is_file() else d.parent
        _require_weights(d, found, repo)
        return repo, d, found
    precision = precision or DEFAULT_PRECISION
    w = d / precision
    if not ((w / "config.json").is_file() and (w / "model.safetensors").is_file()):
        have = [p for p in PRECISIONS if (d / p / "config.json").is_file() and (d / p / "model.safetensors").is_file()]
        msg = f"the {precision} weights are not in {d} (expected {precision}/config.json and {precision}/model.safetensors)."
        if have:
            msg += f" This folder has only: {', '.join(p + '/' for p in have)}; use --precision {have[0]} (precision=\"{have[0]}\"), or"
        msg += f" download them with: hf download {REPO_ID} --include \"{precision}/*\" --local-dir {d}"
        raise FileNotFoundError(msg)
    found = _weights_precision(w)
    if found != precision:
        raise ValueError(f"{w} holds {found} weights, not {precision}")
    _require_weights(w, precision, d)
    return d, w, precision


def verify_repo(repo_dir, weights_dir):
    """manifest check of the shared top-level files and of the selected precision folder only (the other
    precision may be absent). Documentation and records (README.md, figures/, validation/) are skipped as in
    verify_manifest."""
    repo_dir, weights_dir = Path(repo_dir), Path(weights_dir)
    if weights_dir == repo_dir:
        return verify_manifest(repo_dir)
    sub = weights_dir.relative_to(repo_dir).as_posix()
    names = json.loads((repo_dir / "manifest.json").read_text())["files"]
    if not any(n.startswith(sub + "/") for n in names):
        return {"ok": False, "checked": 0, "bad": [], "missing": [f"{sub}/ (not listed in manifest.json)"]}
    res = verify_manifest(repo_dir, only=[n for n in names if "/" not in n] + [sub + "/"])
    res["precision_folder"] = sub + "/"
    return res


def release_compute_dtype(repo_dir, precision):
    """Activation dtype of ``precision`` (release_config.json -> runtime.precisions, else the built-in default)."""
    p = Path(repo_dir) / "release_config.json"
    if p.is_file():
        per = (json.loads(p.read_text()).get("runtime", {}).get("precisions") or {}).get(precision)
        if per is not None and "mlx_compute_dtype" in per:
            return per["mlx_compute_dtype"]
    return VALIDATED_COMPUTE_DTYPE[precision]


class _MLXEngine:
    """Block-causal v2 scoring on mlx-lm (see the MLX backend comment above). Stateless apart from the last
    state cache (kept so consecutive requests with the same state reuse it)."""

    def __init__(self, weights_dir, yes_id, no_id, compute_dtype=None, cache_limit_gib=MLX_CACHE_LIMIT_GIB):
        import mlx.core as mx
        check_mlx_lm()
        from mlx_lm.utils import load_model
        self.mx = mx
        self.cache_limit_bytes = _set_mlx_cache_limit(mx, cache_limit_gib)
        norms = load_norm_sidecar(Path(weights_dir) / NORM_SIDECAR)
        self.model, self.config = load_model(Path(weights_dir))
        if compute_dtype:
            self.model.set_dtype(getattr(mx, compute_dtype))
        self.compute_dtype = compute_dtype
        lm = getattr(self.model, "language_model", self.model)
        self.inner = lm.model
        args = getattr(lm, "args", None) or getattr(self.model, "args", None)
        if not bool(getattr(args, "tie_word_embeddings", True)) or hasattr(lm, "lm_head"):
            raise ValueError("expected tied input/output embeddings")
        self.n_gdn_patched = patch_gdn_l2norm(self.inner)
        self.n_norms_exact = apply_exact_norms(self.inner, norms)
        self.n_full_attention = sum(1 for layer in self.inner.layers if not layer.is_linear)
        w = self.inner.embed_tokens(mx.array([int(yes_id), int(no_id)]))   # dequantised rows if quantised
        w = w.astype(mx.float32)
        self.direction = w[0] - w[1]
        mx.eval(self.direction)
        self._state = None                      # (tuple(prefix ids), cache) of the last state

    def _make_cache(self):
        from mlx_lm.models.cache import ArraysCache
        kv = _block_kv_class()
        return [ArraysCache(size=2) if layer.is_linear else kv() for layer in self.inner.layers]

    def _eval_cache(self, cache):
        arrays = []
        for c in cache:
            if hasattr(c, "keys"):
                if c.keys is not None:
                    arrays += [c.keys, c.values]
            else:
                arrays += [a for a in c.cache if a is not None]
        self.mx.eval(arrays)

    def state_cache(self, prefix, blocks):
        """Cache after the state blocks (computed once; reused while the state is unchanged). -> (cache, reused)"""
        key = tuple(prefix)
        if self._state is not None and self._state[0] == key:
            return self._state[1], True
        self._state = None
        cache = self._make_cache()
        for a, b in blocks:
            self.inner(self.mx.array(prefix[a:b])[None], cache=cache)
            self._eval_cache(cache)
        self._state = (key, cache)
        return cache, False

    def question_scores(self, ids, slots, qblocks, state_cache):
        """Question block(s) on a copy of the state cache -> raw FP32 scores, one per slot (rendered order)."""
        mx = self.mx
        cache = copy_cache(state_cache)
        found = {}
        for i, (a, b) in enumerate(qblocks):
            h = self.inner(mx.array(ids[a:b])[None], cache=cache)[0]
            inside = [(j, s) for j, s in enumerate(slots) if a <= s < b]
            if inside:
                hs = h[mx.array([s - a for _, s in inside])].astype(mx.float32)
                v = hs @ self.direction
                mx.eval(v)
                for (j, _), x in zip(inside, v.tolist()):
                    found[j] = x
            if i + 1 < len(qblocks):
                self._eval_cache(cache)
        if len(found) != len(slots):
            raise RuntimeError("a verdict slot lies outside the question blocks")
        return [float(found[j]) for j in range(len(slots))]

    def reset(self):
        self._state = None

    def close(self):
        self._state = None
        self.mx.clear_cache()


def _check_blocks(r):
    """The protocol invariants a backend relies on: blocks tile [0, n) in order, each <= BLOCK tokens, the
    state ends on a block boundary, every slot lies in a question block."""
    n, p, end = len(r.ids), r.prefix_len, 0
    for a, b in r.blocks:
        if a != end or not a < b <= n or b - a > BLOCK:
            raise ValueError(f"invalid block layout {r.blocks}")
        end = b
    sb, qb = r.state_blocks, r.question_blocks
    if end != n or len(sb) + len(qb) != len(r.blocks) or (sb and sb[-1][1] != p) or not qb:
        raise ValueError(f"invalid block layout {r.blocks} (prefix {p}, {n} tokens)")
    if any(not qb[0][0] <= s < n for s in r.slots):
        raise ValueError("verdict slots must lie in the question blocks")


class JevStyleDecisionMLX(DecisionBase):
    """Apple-silicon runtime on mlx / mlx-lm 0.31.3 (qwen3_5 model code, two numerics corrections).

    ``model_dir``: the repository folder (default: the folder of this script) holding bf16/ and/or 8bit/ next to
    readout_config.json, or one precision folder. ``precision``: "bf16" (default) or "8bit".
    ``temperature``: overrides the calibrated per-type temperatures for every call (default: readout_config.json).
    ``compute_dtype``: "release" (default) = the activation dtype of the chosen precision in release_config.json
    (native bf16 activations for both precisions); "float32" or None (native) override it.
    ``max_len``: total token budget (<= 32,768). ``verify``: re-hash the files in manifest.json before loading.

    The most recent state stays computed: score_many() and consecutive calls / JSONL rows with an identical
    state reuse it (``last_timing["state_reused"]``)."""
    backend = "mlx"

    def __init__(self, model_dir=HERE, precision=None, temperature=None, max_len=CONTEXT_LIMIT,
                 compute_dtype="release", cache_limit_gib=MLX_CACHE_LIMIT_GIB, verify=False):
        repo_dir, weights_dir, precision = resolve_model_dirs(model_dir, precision)
        self.repo_dir, self.weights_dir, self.precision = repo_dir, weights_dir, precision
        if not (repo_dir / "readout_config.json").is_file():
            raise FileNotFoundError(f"readout_config.json (readout + calibration temperature) is not in {repo_dir}; "
                                    f"the runtime does not guess a temperature. Download it with: hf download "
                                    f"{REPO_ID} readout_config.json release_config.json --local-dir {repo_dir}")
        if verify:
            res = verify_repo(repo_dir, weights_dir)
            if not res["ok"]:
                raise RuntimeError(f"integrity check failed: {res}")
        self._setup(repo_dir, weights_dir / "tokenizer.json", max_len, temperature)
        if compute_dtype == "release":
            compute_dtype = release_compute_dtype(repo_dir, precision)
        if compute_dtype not in (None, "float32", "bfloat16", "float16"):
            raise ValueError(f"unsupported compute_dtype {compute_dtype!r}")
        self.engine = _MLXEngine(weights_dir, self.renderer.yes, self.renderer.no, compute_dtype, cache_limit_gib)
        self.compute_dtype = compute_dtype
        self.last_timing = {}

    def _scores_many(self, rendered):
        import time
        for r in rendered:
            _check_blocks(r)
        r0 = rendered[0]
        t0 = time.perf_counter()
        cache, reused = self.engine.state_cache(r0.ids[:r0.prefix_len], r0.state_blocks)
        t1 = time.perf_counter()
        out = [self.engine.question_scores(r.ids, r.slots, r.question_blocks, cache) for r in rendered]
        self.last_timing = {"state_tokens": r0.prefix_len, "state_reused": reused, "state_ms": (t1 - t0) * 1000,
                            "questions": len(rendered), "questions_ms": (time.perf_counter() - t1) * 1000}
        return out

    def close(self):
        self.engine.close()


def main(argv=None):
    ap = base_arg_parser(f"{MODEL_NAME}: typed decisions with MLX (Apple silicon)")
    for a in ap._actions:
        if a.dest == "model_dir":
            a.help = ("repository folder holding bf16/ and/or 8bit/ next to readout_config.json (default: the "
                      "folder of this script), or one precision folder")
    ap.add_argument("--precision", choices=PRECISIONS, default=None,
                    help="weights to load: bf16 (default) or 8bit (affine 8-bit, group 64)")
    ap.add_argument("--compute-dtype", default="release", choices=["release", "float32", "native"],
                    help="activation dtype: the release setting of the precision (release = native), float32, or "
                         "the checkpoint's native dtype")
    ap.add_argument("--cache-limit-gib", type=float, default=MLX_CACHE_LIMIT_GIB)
    args = ap.parse_args(argv)
    cdt = None if args.compute_dtype == "native" else args.compute_dtype
    try:
        engine = JevStyleDecisionMLX(args.model_dir, precision=args.precision, max_len=args.max_len,
                                     compute_dtype=cdt, cache_limit_gib=args.cache_limit_gib, verify=args.verify)
    except (FileNotFoundError, ValueError, MLXVersionError, RuntimeError, ImportError) as e:
        raise SystemExit(f"error: {e}")
    try:
        return run_cli(args, engine)
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
