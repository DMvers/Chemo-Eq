"""Score every search of the paper, LLM and PySR, on its own held-back bins: the tables behind the figures.

    python analysis/score.py <tracks> [--llm-runs ~/chemoeq-runs/paper] [--pysr pysr] [--out results]

A run's answer is its own final equation: the program with the highest search score (validation gain minus the
complexity penalty), with its constants as fitted on train. For the LLM: the seed program and iterations 0-29 of
candidates.jsonl; for PySR: best.py. Split r holds back the same bins in both.

Held-back share (the fit on held-back data): the share of the variation between held-back bin means that the equation
explains, corrected for sampling noise in those means (bins weighted by their segments n):
    1 - (sum n (pred - obs)^2 - noise) / (sum n (obs - mean)^2 - noise),   noise = sum of the bins' segment variances

Writes <out>/:
  bins.csv      per condition and 100 um bin: C (at the bin centre), G, n, and the mean and variance of CEI, speed
                and cos theta
  finals.csv    method, target, setup (x<penalty> or a Fig. 5 setup), scale, run, k (constants), nodes, share,
                gc (CEI: whether the equation follows G/C), code, params
  progress.csv  method, target, run, iteration, share: the held-back share of the best equation so far (x1)
  collapse.csv  method, target, run, cond, bin, C, G, n, observed, predicted: every x1 final equation on every bin
"""
import argparse, json, math, re, sys, warnings
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "llm"), str(ROOT / "llm" / "experiments"), str(ROOT / "pysr")]
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sympy as sp
from chemoeq import data as D
from chemoeq.candidate import Model, check_code, expansion_costs
from chemoeq.variants import experiment
import paper
from likelihood import gain_bits
from search import COMPLEXITY_WEIGHT, PARAM_WEIGHT, to_program

TARGETS = ("cei", "speed", "costheta")
CLIP = {"cei": (-0.995, 0.995), "costheta": (-0.995, 0.995), "speed": (0.05, 1000.0)}
PYSR_SCALES = {"cei": ["1", "0_01", "0_1", "10", "20", "30", "100"], "speed": ["1", "0_01", "0_1", "10", "20", "30", "100"],
               "costheta": ["1"]}
NUM = r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?"


def share(pred, y, var, n):
    noise = np.sum(np.where(n > 1, var, 0.0))
    return 1 - (np.sum(n * (pred - y) ** 2) - noise) / (np.sum(n * (y - np.sum(n * y) / n.sum()) ** 2) - noise)


def bin_table(df):
    """Per condition and 100 um bin: centre C, G, n and each target's mean and variance."""
    d = df.assign(bin=np.floor(df["x_mid"].to_numpy(float) / 100).astype(int), cmin=df["C"] - df["G"] * df["x_mid"])
    g = d.groupby(["cond", "bin"], observed=True, sort=True)
    B = g.agg(G=("G", "first"), cmin=("cmin", "mean"), n=("G", "size")).reset_index()
    for t in TARGETS:
        B[t], B[t + "_var"] = g[t].mean().to_numpy(), g[t].var().fillna(0).to_numpy()
    B["cond"] = B["cond"].astype(str)
    B["C"] = np.maximum(B.cmin + B.G * (B["bin"] + 0.5) * 100, 0.0)
    return B.drop(columns="cmin")


def follows_gc(f):
    """Whether a CEI equation follows G/C, however it is written: at 25 concentrations from 10 nM to 10 uM, the gradient
    at which the prediction first reaches half its maximum over 1e-4..1e4 nM/um (where that maximum exceeds 0.05);
    the slope of its log against log C is 1 for an equation in G/C and 0 for one in G alone. Follows G/C: slope > 0.5."""
    cs, gs = np.geomspace(10, 1e4, 25), np.geomspace(1e-4, 1e4, 600)
    CC, GG = np.meshgrid(cs, gs)
    with np.errstate(all="ignore"):
        M = np.broadcast_to(np.asarray(f(CC.ravel(), GG.ravel()), float), CC.size).reshape(GG.shape)
    lc, lg = [], []
    for j, c in enumerate(cs):
        mx = np.nanmax(M[:, j])
        k = int(np.argmax(M[:, j] >= 0.5 * mx)) if np.isfinite(mx) and mx >= 0.05 else 0
        if k:
            lc.append(np.log(c)); lg.append(np.log(gs[k]))
    return len(lc) >= 5 and np.polyfit(lc, lg, 1)[0] > 0.5


def snapshots(log: Path):
    """PySR's printed hall of fame, about every 10 iterations: [(iteration, [equation, ...]), ...]."""
    out, it, entries, inside = [], None, [], False
    for ln in log.read_text().splitlines():
        m = re.match(r"Progress: (\d+) / \d+ total iterations", ln)
        if m:
            it = int(m.group(1)); continue
        if ln.startswith("Complexity  Loss"):
            entries, inside = [], True; continue
        if not inside:
            continue
        if ln.startswith("───"):
            out.append((it, entries)); inside = False; continue
        m = re.match(rf"^(\d+)\s+({NUM})\s+({NUM})\s+y = (.*)$", ln)
        if m:
            entries.append(m.group(4))
        elif entries and ln.startswith(" "):                       # a wrapped equation continues
            e = entries[-1]
            entries[-1] = (e[:-3] if e.endswith("...") else e) + ln.strip().removesuffix("...")
    return [s for s in out if s[0] is not None and s[1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--llm-runs", type=Path, default=Path("~/chemoeq-runs/paper").expanduser())
    ap.add_argument("--pysr", type=Path, default=ROOT / "pysr", help="the pysr folder (its runs/ and data/)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    settings = {**paper.SETTINGS, "data": replace(paper.SETTINGS["data"], root=str(a.tracks.resolve()))}
    exp = {v: experiment(paper.VARIANTS, settings, v, 0, "score", out_dir=str(a.out / "tmp")) for v in paper.VARIANTS}
    df = D.load_for(exp["cei"], log=lambda *x: None)["df"]
    df = df.assign(cei=D.arrays(df, exp["cei"].inputs)["cei"], costheta=np.cos(df["theta"].to_numpy(float)))
    B = bin_table(df)
    B.to_csv(a.out / "bins.csv", index=False)
    X = {"C": B.C.to_numpy(float), "G": B.G.to_numpy(float), "theta": np.zeros(len(B))}
    conds = sorted(B.cond.unique())
    n_blocks = math.ceil(settings["data"].kept_range()[1] / 100)
    test = [np.array([blk[c][b] == "test" for c, b in zip(B.cond, B["bin"])])
            for blk in (D.draw_blocks(conds, n_blocks, settings["test_fraction"], settings["val_fraction"], s)
                        for s in paper.SPLIT_SEEDS)]

    def held_share(t, r, pred):
        h = test[r]
        return share(pred[h], B[t].to_numpy()[h], B[t + "_var"].to_numpy()[h], B.n.to_numpy(float)[h])

    def llm_pred(variant, code, p):
        return Model(code, exp[variant].args, exp[variant].likelihood).mean_cos(X, np.asarray(p, float))

    def pysr_pred(t, code, p):
        ns = {"np": np, "math": math}
        exec(code, ns)
        with np.errstate(all="ignore"):
            out = np.broadcast_to(np.asarray(ns["model"](X["C"], X["G"], np.asarray(p, float)), float), len(B))
        return np.clip(out, *CLIP[t])

    finals, prog, coll = [], [], []

    def record(method, t, setup, scale, r, code, p, pred):
        meta = check_code(code, ["C", "G"], 99, expansion_costs())
        gc = follows_gc(_model(code, p)) if t == "cei" else None
        finals.append(dict(method=method, target=t, setup=setup, scale=scale, run=r, k=meta["n_params"],
                           nodes=meta["complexity"], share=held_share(t, r, pred), gc=gc, code=code,
                           params=json.dumps(list(map(float, p)))))
        if setup == "x1":
            coll.append(B[["cond", "bin", "C", "G", "n"]].assign(method=method, target=t, run=r, observed=B[t],
                                                                 predicted=pred))

    # ---------------------------------------------------------------- LLM
    for variant in paper.VARIANTS:
        t = variant.split("_")[0]
        rest = variant[len(t) + 1:]
        setup = "x1" if not rest else (f"x{rest[1:].replace('_', '.')}" if rest.startswith("x") else rest)
        scale = float(setup[1:]) if setup.startswith("x") else 1.0
        for r in range(len(paper.SPLIT_SEEDS)):
            run = a.llm_runs / f"{variant}_r{r}"
            if not (run / "candidates.jsonl").exists():
                continue
            assert json.loads((run / "split.json").read_text())["seed"] == paper.SPLIT_SEEDS[r], run
            c = [d for d in map(json.loads, open(run / "candidates.jsonl")) if d.get("ok") and d.get("iteration", -1) < 30]
            best = max(c, key=lambda d: d["score"])
            record("llm", t, setup, scale, r, best["code"], best["params"], llm_pred(variant, best["code"], best["params"]))
            if setup == "x1":                                      # the best equation so far, at every iteration
                for it in range(30):
                    b = max((d for d in c if d["iteration"] <= it), key=lambda d: d["score"])
                    prog.append(dict(method="llm", target=t, run=r, iteration=it + 1,
                                     share=held_share(t, r, llm_pred(variant, b["code"], b["params"]))))

    # ---------------------------------------------------------------- PySR
    for t, scales in PYSR_SCALES.items():
        for s in scales:
            scale = float(s.replace("_", "."))
            for r in range(len(paper.SPLIT_SEEDS)):
                run = a.pysr / "runs" / f"{t}_x{s}_r{r}"
                if not (run / "best.py").exists():
                    continue
                text = (run / "best.py").read_text()
                ns = {}
                exec(text, ns)
                code, p = text[text.index("N_PARAMS"):], ns["FITTED_PARAMS"]
                record("pysr", t, f"x{scale:g}", scale, r, code, p, pysr_pred(t, code, p))
                if s != "1":
                    continue
                # what search.py would have answered at each printed hall of fame, then its answer
                rows = pd.read_csv(a.pysr / "data" / f"bins_r{r}.csv")
                val = rows[rows["set"] == "val"]
                for it, eqs in snapshots(run.parent / f"{run.name}.log"):
                    pick = None
                    for eq in eqs:
                        try:
                            code_i, p_i = to_program(sp.sympify(eq.replace("^", "**")))
                            nodes = check_code(code_i, ["C", "G"], 99, expansion_costs())["complexity"]
                            g = gain_bits(val[t].to_numpy(), _model(code_i, p_i)(val.C.to_numpy(float), val.G.to_numpy(float)), t)
                        except Exception:
                            continue
                        sc_ = g - (PARAM_WEIGHT * len(p_i) + COMPLEXITY_WEIGHT * nodes)
                        if np.isfinite(sc_) and (pick is None or sc_ > pick[0]):
                            pick = (sc_, code_i, p_i)
                    if pick and it < 100:
                        prog.append(dict(method="pysr", target=t, run=r, iteration=it,
                                         share=held_share(t, r, pysr_pred(t, pick[1], pick[2]))))
                prog.append(dict(method="pysr", target=t, run=r, iteration=100, share=finals[-1]["share"]))

    F = pd.DataFrame(finals)
    F.to_csv(a.out / "finals.csv", index=False)
    pd.DataFrame(prog).to_csv(a.out / "progress.csv", index=False)
    pd.concat(coll, ignore_index=True).to_csv(a.out / "collapse.csv", index=False)
    pd.set_option("display.width", 200)
    print(F.pivot_table(index=["target", "method"], columns="setup", values="share", aggfunc="median").round(3).to_string())
    print(F.pivot_table(index=["target", "method"], columns="setup", values="share", aggfunc="count").to_string())


def _model(code, p):
    ns = {"np": np, "math": math}
    exec(code, ns)
    p = np.asarray(p, float)
    return lambda c, g: np.broadcast_to(np.asarray(ns["model"](c, g, p), float), np.shape(c))


if __name__ == "__main__":
    main()
