"""The proposal-fit-score loop (LLM-SR / FunSearch style, island model).

    search.run(exp)        # exp: an Experiment (see experiments/)
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import shutil
import socket
from dataclasses import asdict, is_dataclass
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import data as D
from . import records
from .equation import equation
from .candidate import CandidateError, Evaluator, extract_code, extract_idea
from .experiment import input_summary


# ------------------------------------------------------------------ LLM client

class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP over a Unix domain socket (a file on disk) instead of a TCP port."""

    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.socket_path)
        self.sock = s


def default_socket_path() -> str:
    """Must match SOCK in scripts/serve_qwen.sh. Kept in the home directory rather
    than $XDG_RUNTIME_DIR, which systemd deletes when you log out even if the
    server is still running in the background."""
    return os.path.expanduser("~/.chemoeq/vllm.sock")


class Reply(str):
    """The model's answer (what is parsed for the idea and the code), carrying what else
    the server said: its reasoning, why it stopped ("stop", or "length" when it reached
    max_tokens) and how many tokens it wrote, reasoning included. With tools: the calls it
    made (OpenAI format) and the length of the prompt it answered."""

    def __new__(cls, text, reasoning="", finish_reason=None, tokens=None, tool_calls=None, prompt_tokens=None):
        obj = super().__new__(cls, text)
        obj.reasoning, obj.finish_reason, obj.tokens = reasoning, finish_reason, tokens
        obj.tool_calls, obj.prompt_tokens = list(tool_calls or []), prompt_tokens
        return obj


class LocalLLM:
    """Minimal OpenAI-compatible chat client (vLLM `serve`, llama.cpp server, SGLang...).

    With LLM(uds="auto") (the default), requests go over a Unix domain socket and no
    network port is involved; LLM(uds=None, base_url=...) uses TCP instead.
    """

    def __init__(self, cfg):
        self.cfg = cfg = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
        uds = cfg.get("uds")
        self.uds = os.path.expanduser(default_socket_path() if uds == "auto" else uds) if uds else None

    def __call__(self, messages: list[dict], seed: int | None = None, tools: list | None = None,
                 tool_choice: str | None = None, max_tokens: int | None = None) -> str:
        c = self.cfg
        body = {"model": c["model"], "messages": messages, "temperature": c["temperature"],
                "top_p": c["top_p"], "max_tokens": max_tokens or c["max_tokens"]}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        for k in ("top_k", "min_p", "presence_penalty", "repetition_penalty"):
            if c.get(k) is not None:            # vLLM takes these next to the OpenAI ones
                body[k] = c[k]
        if seed is not None:
            body["seed"] = seed
        if c.get("disable_thinking"):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif c.get("reasoning_effort"):
            body["chat_template_kwargs"] = {"reasoning_effort": c["reasoning_effort"]}
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {c['api_key']}"}
        if self.uds:
            conn = _UnixHTTPConnection(self.uds, timeout=c["timeout_s"])
            try:
                conn.request("POST", "/v1/chat/completions", body=data, headers=headers)
                r = conn.getresponse()
                raw = r.read()
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {raw[:300]!r}")
                out = json.loads(raw)
            finally:
                conn.close()
        else:
            req = urllib.request.Request(c["base_url"].rstrip("/") + "/chat/completions",
                                         data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=c["timeout_s"]) as r:
                out = json.load(r)
        choice = out["choices"][0]
        msg = choice["message"]
        # with the server's --reasoning-parser, the reasoning comes apart from the answer
        # ("reasoning" or, in older vLLM, "reasoning_content"); without it, it is in the
        # answer as <think>...</think>, and is taken out here
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if "</think>" in content:
            head, content = content.split("</think>", 1)
            reasoning = reasoning or head.replace("<think>", "").strip()
            content = content.strip()
        usage = out.get("usage") or {}
        return Reply(content, reasoning=reasoning, finish_reason=choice.get("finish_reason"),
                     tokens=usage.get("completion_tokens"), tool_calls=msg.get("tool_calls"),
                     prompt_tokens=usage.get("prompt_tokens"))

    def check(self) -> str:
        """Ask the server which models it serves; raises if it isn't reachable."""
        if self.uds:
            if not os.path.exists(self.uds):
                raise RuntimeError(f"no socket at {self.uds}: is the model server running? Start it with: bash scripts/prelaunch.sh")
            conn = _UnixHTTPConnection(self.uds, timeout=10)
            try:
                conn.request("GET", "/v1/models")
                raw = conn.getresponse().read()
            finally:
                conn.close()
        else:
            with urllib.request.urlopen(self.cfg["base_url"].rstrip("/") + "/models", timeout=10) as r:
                raw = r.read()
        ids = [m["id"] for m in json.loads(raw).get("data", [])]
        if self.cfg["model"] not in ids:
            raise RuntimeError(f"server is up but serves {ids}, not {self.cfg['model']!r}: fix llm=LLM(model=...) in the experiment")
        return ", ".join(ids)


# ------------------------------------------------------------------ islands

class Islands:
    """Code to support several independent populations (islands), but we used only one island in our experiments
    """

    def __init__(self, n, size, rng):
        self.pop = [[] for _ in range(n)]
        self.size, self.rng = size, rng

    def add(self, i, entry):
        """Add unless an identical program, or one that behaves the same and is no
        more complex, is already there. A simpler program that behaves the same
        replaces the more complex one."""
        isl = self.pop[i]
        fp = entry.get("fingerprint")
        for j, e in enumerate(isl):
            if e["hash"] == entry["hash"]:
                return False
            if fp and e.get("fingerprint") == fp:
                if e["complexity"] <= entry["complexity"]:
                    return False
                del isl[j]
                break
        isl.append(entry)
        isl.sort(key=lambda e: -e["score"])
        del isl[self.size:]
        return entry in isl

    def sample(self, i, k, rank_temperature):
        """Draw k distinct examples, weighted by rank (best = rank 0)."""
        isl = self.pop[i]                       # kept sorted best first
        if len(isl) <= k:
            return list(isl)
        w = np.exp(-np.arange(len(isl)) / max(rank_temperature, 1e-9))
        idx = self.rng.choice(len(isl), size=k, replace=False, p=w / w.sum())
        return [isl[j] for j in idx]

    def best(self):
        allv = [e for isl in self.pop for e in isl]
        return max(allv, key=lambda e: e["score"]) if allv else None

    def reset_weakest(self):
        bests = [max((e["score"] for e in isl), default=-math.inf) for isl in self.pop]
        order = np.argsort(bests)
        weak, strong = order[: len(order) // 2], order[len(order) // 2:]
        for w in weak:
            donor = self.pop[int(self.rng.choice(strong))]
            self.pop[w] = [dict(donor[0])] if donor else []

    def state(self):
        return self.pop


# ------------------------------------------------------------------ loop

def search_arrays(exp, splits):
    return {"fit": D.arrays(D.subsample(splits["train"], exp.fit_rows_per_condition, exp.random_seed), exp.inputs),
            "val": D.arrays(D.subsample(splits["val"], exp.val_rows_per_condition, exp.random_seed + 1), exp.inputs)}


def feedback_data(exp, splits, conditions):
    """For diagnostics_by="concentration": the rows behind the feedback table (arrays with
    the bin's name as `cond`), the bins in display order, and each bin's condition.
    None for the usual table with one line per condition."""
    if exp.diagnostics_by != "concentration" or not exp.show_diagnostics:
        return None
    rows, order, owner = D.feedback_bins(splits["train"], conditions, exp.diagnostics_bins, exp.diagnostics_span,
                                         exp.diagnostics_rows_per_bin, exp.diagnostics_min_rows, exp.random_seed + 2,
                                         x_range_um=exp.data.x_range_um)
    return D.arrays(rows, exp.inputs), order, owner


def _probe(train):
    """Fixed probe set for behavioural de-duplication."""
    rng = np.random.default_rng(12345)
    n_all = train["theta"].size
    idx = rng.choice(n_all, min(2000, n_all), replace=False)
    # arrays with one entry per row are sampled; others (the cell centres of full-distribution fits) kept whole
    return {k: v[idx] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n_all else v for k, v in train.items()}


def _snapshot(exp, out: Path, log):
    """Keep a copy of the experiment file with the run (and earlier versions if it changed)."""
    if not exp.source or not Path(exp.source).is_file():
        return
    dst = out / "experiment.py"
    new = Path(exp.source).read_text()
    if dst.exists() and dst.read_text() != new:
        old = out / f"experiment.{time.strftime('%Y%m%d-%H%M%S', time.localtime(dst.stat().st_mtime))}.py"
        shutil.copyfile(dst, old)
        log(f"the experiment file changed since the last start; previous version kept as {old.name}")
    dst.write_text(new)


def run(exp, llm=None, log=print):
    out = exp.run_dir
    out.mkdir(parents=True, exist_ok=True)
    _snapshot(exp, out, log)
    log(f"results go to {out}")
    rng = np.random.default_rng(exp.random_seed)
    prepared = D.load_for(exp, log)
    splits = D.split(prepared["df"], exp, log)
    A = search_arrays(exp, splits)
    log(f"{exp.name}: {exp.signature}; rows used for search: fit {D.segments(A['fit']):,}  "
        f"val {D.segments(A['val']):,} (full train {D.segments(splits['train']):,}, "
        f"test {D.segments(splits['test']):,} held back)")
    if exp.bin_rows_um is not None:
        log(f"bin rows: every set summarised into {exp.bin_rows_um:g} um bins at their centre's concentration: "
            f"fit {len(splits['train'])} rows, val {len(splits['val'])}, test {len(splits['test'])}")
    if exp.bin_weights == "equal":
        log("bin rows count equally (weight 1 each, whatever their number of segments)")
    if exp.split_unit == "condition":
        for k, label in (("train", "fit"), ("val", "val"), ("test", "test")):
            log(f"  {label} conditions: {', '.join(sorted(splits[k]['cond'].unique()))}")
    feedback = feedback_data(exp, splits, prepared["conditions"])
    summary = input_summary(exp, A["fit"], len(prepared["conditions"]), feedback)
    if feedback:
        log(f"feedback table: {len(feedback[1])} concentration bins, {D.segments(feedback[0]):,} fitting segments")
    if llm is None:
        llm = LocalLLM(exp.llm)
        log(f"LLM server OK: {llm.check()}")
    ev = Evaluator(exp.scoring(), A["fit"], A["val"], probe=_probe(A["fit"]),
                   diagnostics=exp.show_diagnostics, workers=exp.fit_workers,
                   memory_gb=exp.fit_memory_gb, diag_data=feedback[0] if feedback else None)
    islands = Islands(exp.islands, exp.island_size, rng)
    log_path = out / "candidates.jsonl"
    start_iter = 0
    if log_path.exists() and (not log_path.with_suffix(".csv").exists()
                              or not records.header_ok(log_path.with_suffix(".csv"))):
        records.write_csv(log_path, outputs=exp.likelihood.outputs)    # missing, or older columns

    ckpt = out / "islands.json"
    held = {k: sorted(splits[k]["cond"].unique()) for k in ("test", "val")}
    if exp.split_unit != "condition":                   # the same conditions in every set: the blocks decide
        held["blocks"] = json.loads((out / "split.json").read_text())["blocks"]
    run_id = exp.fingerprint() + "-" + hashlib.sha1(json.dumps(held).encode()).hexdigest()[:10]
    if ckpt.exists():                                   # resume
        st = json.loads(ckpt.read_text())
        if st.get("run_id") != run_id:
            raise RuntimeError(
                f"{ckpt} was written with different data, inputs, split, likelihood or scoring. "
                "Give the experiment a new name (or out_dir), or delete the checkpoint to start over.")
        islands.pop, start_iter = st["islands"], st["iteration"] + 1
        rng.bit_generator.state = st["rng"]
        log(f"resumed at iteration {start_iter}")
    else:
        results = ev.run(list(exp.seeds))
        bad = [f"seed {i}: {r.get('error')}" for i, r in enumerate(results) if not r["ok"]]
        if bad:
            ev.close()
            raise RuntimeError("seed program(s) failed:\n  " + "\n  ".join(bad))
        for i, r in enumerate(results):
            r.update(iteration=-1, island=-1, kept=True, idea=f"seed program {i} (given)")
            for k in range(exp.islands):
                islands.add(k, dict(r))
            _append(log_path, r, exp)
            log(f"seed {i}: val gain {r['val_gain']:.4f} bits/segment")

    # Generation and fitting overlap: while the worker processes fit the programs
    # of iteration k, the LLM already writes the proposals of iteration k + 1.
    # This speeds things up a lot, but does lead to a lag of one iteration: prompts for k + 1 draw on the population
    # as it was before k's results arrived (as in asynchronous FunSearch).
    # overlap=False turns this off: slower, but every prompt sees all results so far.
    pool = ThreadPoolExecutor(max_workers=exp.batch)
    pending = None                       # iteration whose fits are still running
    last_done = [time.time()]            # when the previous iteration finished

    def finish(job):
        """Collect one iteration's fits, update the islands, checkpoint, log."""
        k, keep, ideas, slots, n_replies, handle, t0, t_gen, fits = job
        t_wait = time.time()
        results = ev.collect(handle)
        t_wait = time.time() - t_wait
        n_ok = n_new = 0
        for isl, idea, slot, r in zip(keep, ideas, slots, results):
            r.update(iteration=k, island=isl, prompt=slot, idea=idea)
            if fits[slot] is not None:           # fit_model calls made before this answer
                r["fits"] = fits[slot]
            r["kept"] = bool(r["ok"] and islands.add(isl, r))
            n_ok += bool(r["ok"]); n_new += r["kept"]
            _append(log_path, r, exp)
        if exp.reset_every and (k + 1) % exp.reset_every == 0:
            islands.reset_weakest()
        _atomic_write(ckpt, json.dumps({"iteration": k, "islands": islands.state(),
                                        "rng": rng.bit_generator.state, "run_id": run_id}))
        b = islands.best()
        log(f"iter {k:4d}  valid {n_ok}/{n_replies}  new {n_new}  "
            f"best score {b['score']:.4f} (val {b['val_gain']:.4f}, k={b['n_params']})  "
            + f"{time.time() - last_done[0]:.0f}s (LLM {t_gen:.0f}s, then waited {t_wait:.0f}s for fits)")
        last_done[0] = time.time()

    try:
        for it in range(start_iter, exp.iterations):
            if pending and not exp.overlap:      # wait, so this prompt sees iteration it - 1
                finish(pending); pending = None
            t0 = time.time()
            picks = [int(rng.integers(exp.islands)) for _ in range(exp.batch)]
            msgs = [exp.render(islands.sample(i, exp.examples_per_prompt, exp.rank_temperature),
                               summary, prepared["conditions"]) for i in picks]
            seeds = [int(rng.integers(1 << 30)) for _ in msgs]
            for attempt in range(exp.llm.retries + 1):
                replies = list(pool.map(lambda a: _safe_call(llm, *a), zip(msgs, seeds)))
                if not all(r.startswith(LLM_ERROR) for r in replies):
                    break
                if attempt == exp.llm.retries:
                    if pending:                  # keep the work already done
                        finish(pending); pending = None
                    raise RuntimeError(f"LLM server failed {attempt + 1} batches in a row "
                                       f"(last error: {replies[0][:300]}). Progress is saved up to "
                                       f"iteration {it - 1}; restart the server and rerun to resume.")
                wait = min(300, 15 * 2 ** attempt)
                log(f"iter {it}: all LLM requests failed ({replies[0][:120]}); retrying in {wait}s")
                time.sleep(wait)
            t_gen = time.time() - t0
            if exp.save_prompts:                 # what the model got and what it answered
                with open(out / "prompts.jsonl", "a") as f:
                    for j, (isl, m, sd, rep) in enumerate(zip(picks, msgs, seeds, replies)):
                        rec = {"iteration": it, "prompt": j, "island": isl, "seed": sd, "messages": m, "reply": str(rep)}
                        if getattr(rep, "reasoning", ""):
                            rec["reasoning"] = rep.reasoning
                        for k in ("finish_reason", "tokens", "fits", "conversation", "thoughts"):
                            if getattr(rep, k, None) is not None:
                                rec[k] = getattr(rep, k)
                        f.write(json.dumps(rec) + "\n")
            codes, keep, ideas, slots = [], [], [], []
            for j, (isl, rep) in enumerate(zip(picks, replies)):
                try:
                    codes.append(extract_code(rep)); keep.append(isl); slots.append(j)
                    ideas.append(extract_idea(rep))
                except CandidateError as e:
                    err = str(e)
                    if getattr(rep, "finish_reason", None) == "length":
                        err = f"reply cut off at max_tokens ({getattr(rep, 'tokens', None) or exp.llm.max_tokens} tokens): {err}"
                    elif getattr(rep, "finish_reason", None) == "context":
                        err = f"conversation reached the server's context ({exp.llm.context_tokens} tokens): {err}"
                    rec = {"ok": False, "error": err, "raw": rep[:2000], "iteration": it, "island": isl,
                           "prompt": j, "kept": False, "idea": extract_idea(rep)}
                    if getattr(rep, "fits", None) is not None:
                        rec["fits"] = rep.fits
                    _append(log_path, rec, exp)
            if pending:
                finish(pending)
            pending = (it, keep, ideas, slots, len(replies), ev.submit(codes, seed=it * 1000), t0, t_gen,
                       [getattr(rep, "fits", None) for rep in replies])
        if pending:
            finish(pending); pending = None
    finally:
        ev.close()
        pool.shutdown()
    best = dict(islands.best())
    write_best(out / "best.py", best, exp)
    records.write_csv(log_path, outputs=exp.likelihood.outputs)   # full rebuild: consistent after resumes
    n_fit = D.segments(A["fit"])
    log(f"normal fit ({len(set(A['fit']['cond']))} fitting conditions, {n_fit:,} segments): "
        f"train gain {best['train_gain']:.4f}, val gain {best['val_gain']:.4f} bits/segment; "
        f"parameters {_fmt(best['params'])}")
    if exp.final_full_fit:
        try:
            best["full_fit"] = full_fit(exp, best, splits, log)
        except Exception as e:                   # the search result stands without it
            log(f"fit on all data failed ({type(e).__name__}: {e})")
        full = best.get("full_fit")
        if full:
            write_best(out / "best_full.py", {**best, "params": full["params"], "train_gain": full["gain"]},
                       exp, fitted_on=f"all {full['conditions']} conditions, {full['segments']:,} segments")
    return best


def _fmt(params) -> str:
    return "[" + ", ".join(f"{v:.5g}" for v in params) + "]"


def full_fit(exp, best: dict, splits: dict, log=print) -> dict | None:
    """The best program refitted on every condition and every segment (test conditions
    included), in a worker process like any other fit. For looking at, after the search:
    nothing here feeds back into which program was chosen."""
    import pandas as pd
    from .evaluate import _run_jobs
    whole = D.arrays(pd.concat([splits["train"], splits["val"], splits["test"]]), exp.inputs)
    shared = dict(args=exp.args, likelihood=exp.likelihood, max_params=exp.max_params,
                  fit_restarts=exp.eval_fit_restarts, fit_maxiter=exp.eval_fit_maxiter,
                  train=whole, test=whole, ctrl_mask=np.zeros(whole["theta"].size, bool))
    r = _run_jobs([("best", best["code"], 0)], shared, 1, exp.eval_fit_timeout_s,
                  lambda *a: None, exp.fit_memory_gb)[0]
    if not r["ok"]:
        log(f"fit on all data failed ({r['error']})")
        return None
    res = {"params": r["params"], "gain": r["refit_gain"], "conditions": len(set(whole["cond"])),
           "segments": D.segments(whole), "per_condition": r["per_condition"]}
    log(f"fit on all data ({res['conditions']} conditions, {res['segments']:,} segments): "
        f"gain {res['gain']:.4f} bits/segment; parameters {_fmt(res['params'])}")
    return res


def write_best(path: Path, best: dict, exp, fitted_on: str | None = None):
    """best.py: the program with its fitted parameters, equation and scores in the
    header, and the parameters also as FITTED_PARAMS (model(..., FITTED_PARAMS) works)."""
    p, outs = best["params"], exp.likelihood.outputs
    lines = [
        f"# best program by validation score: val gain {best['val_gain']:.4f} bits/segment, "
        f"fitted params {p}",
        f"# experiment {exp.name}; likelihood {exp.likelihood!r}",
        f"# score {best['score']:.6f} (val gain {best['val_gain']:.6f} minus complexity penalty), "
        f"train gain {best['train_gain']:.6f} bits/segment",
        f"# found in iteration {best.get('iteration')}, island {best.get('island')}; "
        f"{best['n_params']} parameters, {best['complexity']} syntax-tree nodes",
        f"# fitted parameters (maximum likelihood on {fitted_on or 'the training conditions'}):",
        *[f"#   p[{i}] = {v:.10g}" for i, v in enumerate(p)],
        "# equation:",
        *[f"#   {l}" for l in equation(best["code"], p, outputs=outs).splitlines()],
        "# with the fitted values:",
        *[f"#   {l}" for l in equation(best["code"], p, numeric=True, outputs=outs).splitlines()],
        "# Written by the LLM: read it before running it outside chemoeq.",
        "import math", "", "import numpy as np", "",
        f"FITTED_PARAMS = [{', '.join(repr(float(v)) for v in p)}]", "",
        best["code"], "",
    ]
    path.write_text("\n".join(lines))


def describe_best(best: dict, outputs=("mu", "kappa")) -> str:
    """Text shown at the end of a search: scores, fitted parameters, equation, code."""
    return (f"best program (iteration {best.get('iteration')}, island {best.get('island')}):\n"
            f"  score {best['score']:.6f}   val gain {best['val_gain']:.6f}   "
            f"train gain {best['train_gain']:.6f} bits/segment\n"
            "fitted parameters:\n"
            + "".join(f"  p[{i}] = {v:.10g}\n" for i, v in enumerate(best["params"]))
            + "equation:\n" + _indent(equation(best["code"], best["params"], outputs=outputs))
            + "with the fitted values:\n"
            + _indent(equation(best["code"], best["params"], numeric=True, outputs=outputs))
            + _describe_full(best, outputs)
            + "\ncode:\n" + best["code"])


def _describe_full(best: dict, outputs) -> str:
    """The same program fitted on all data, if the search ended with that fit."""
    full = best.get("full_fit")
    if not full:
        return ""
    return (f"fitted on all data ({full['conditions']} conditions, {full['segments']:,} segments): "
            f"gain {full['gain']:.6f} bits/segment\n"
            + "".join(f"  p[{i}] = {v:.10g}\n" for i, v in enumerate(full["params"]))
            + "with these values:\n"
            + _indent(equation(best["code"], full["params"], numeric=True, outputs=outputs)))


def _indent(text, pad="  "):
    return "".join(pad + l + "\n" for l in text.splitlines())


LLM_ERROR = "LLM error: "


def _safe_call(llm, messages, seed):
    try:
        return llm(messages, seed=seed)
    except Exception as e:  # server hiccup: count as an invalid proposal
        return f"{LLM_ERROR}{type(e).__name__}: {e}"


def _atomic_write(path: Path, text: str):
    """Write then rename, so a crash mid-write never leaves a corrupt checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append(path, rec, exp):
    rec = {k: v for k, v in rec.items() if k != "traceback"}
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    records.append(Path(path).with_suffix(".csv"), rec, outputs=exp.likelihood.outputs)
