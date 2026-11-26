"""Worker processes for the final refit of a search's best program on all its data (search.full_fit)."""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback

import numpy as np

from .candidate import (Model, _limit_worker, check_code, fit, memory_limit_bytes,
                        per_condition)
from .likelihood import gain_bits

_S = {}


def _init(shared, mem_bytes=None):
    _S.clear()
    _S.update(shared)
    _limit_worker(mem_bytes)          # no file writes, memory cap: as in search


def _refit_job(args):
    """Fit one program on the refit set and score it on test (runs in a worker)."""
    name, code, seed = args
    S = _S
    try:
        meta = check_code(code, S["args"], max(S["max_params"], 12), S.get("op_costs"))
        model = Model(code, S["args"], S["likelihood"])
        p, tr_nll = fit(model, S["train"], meta["init"], S["fit_restarts"],
                        np.random.default_rng(seed), S["fit_maxiter"])
        lp = model.logpdf(S["test"], p)
        pred = model.mean_cos(S["test"], p)
        if not np.all(np.isfinite(lp)):
            raise ValueError("non-finite likelihood on test conditions")
        ctrl = S["ctrl_mask"]
        w = S["test"].get("weight", np.ones(len(lp)))        # bin rows: weighted by their segments
        return {"name": name, "ok": True, "code": code, "params": p.tolist(),
                "n_params": meta["n_params"], "complexity": meta["complexity"],
                "refit_gain": gain_bits(-tr_nll, S["likelihood"]), "lp": lp, "w": w,
                # Net bias in no-gradient controls, predicted vs observed. A sound signal
                # law predicts ~0.
                "control_pred_cos": float(np.average(pred[ctrl], weights=w[ctrl]))
                if ctrl.any() else None,
                "control_obs_cos": float(np.average(model.lik.observed(S["test"])[ctrl], weights=w[ctrl]))
                if ctrl.any() else None,
                "per_condition": per_condition(model, S["test"], p)}
    except Exception as e:
        return {"name": name, "ok": False, "code": code,
                "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=3)}


def _run_jobs(jobs, shared, n, timeout, log, memory_gb="auto"):
    ctx = mp.get_context("forkserver")
    mem = memory_limit_bytes(memory_gb, n)
    pool = ctx.Pool(n, initializer=_init, initargs=(shared, mem))
    start = time.time()
    handles = [pool.apply_async(_refit_job, (j,)) for j in jobs]
    out = []
    for i, (j, h) in enumerate(zip(jobs, handles)):
        try:
            out.append(h.get(timeout=max(1.0, start + timeout * (i // n + 1) - time.time())))
        except mp.TimeoutError:
            out.append({"name": j[0], "ok": False, "code": j[1], "error": f"refit timed out (> {timeout}s)"})
        log(f"  {out[-1]['name']}: " + ("ok" if out[-1]["ok"] else out[-1]["error"]))
    pool.terminate()
    return out


