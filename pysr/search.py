"""One PySR search:

    python pysr/search.py <target> <penalty scale> <split> [data folder, default pysr/data]

Results: pysr/runs/<target>_x<scale>_r<split>/ (best.py: the answer; hall_of_fame.csv; PySR's own files). Keep the
search's printed log (run.sh writes it next to the folder): its hall-of-fame snapshots give the per-iteration curves.

  1. PySR evolves expressions in C (nM) and G (nM/um) and fits their constants on the train bins, minimising minus
     the log likelihood of each bin's mean (likelihood.JULIA_LOSS) plus the penalty: PySR's sizes are counted in
     the LLM search's units (a node 1, a variable 2, a constant 5 nodes plus its parameter's 10, a binary operator 2,
     exp and log 5, tanh 5 + 12 for its expansion into exp), and parsimony = scale x 1e-5 ln 2 / baseline (PySR's
     cost is loss / baseline + parsimony x size, baseline the loss of the mean).
  2. Every hall-of-fame expression (the best at each size) is written as a program model(C, G, p) with its constants
     as p, and scored as the LLM search scores a proposal: val gain minus scale x (1e-4 per constant + 1e-5 per
     node, nodes counted by the LLM search's own code). The best score is the answer.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from likelihood import JULIA_LOSS, gain_bits, pysr_loss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "llm"))
from chemoeq.candidate import check_code, expansion_costs        # noqa: E402

PARAM_WEIGHT, COMPLEXITY_WEIGHT = 1e-4, 1e-5                     # the penalty at x1
BINARY, UNARY = ["+", "-", "*", "/", "^"], ["exp", "log", "tanh"]
COSTS = {"+": 2, "-": 2, "*": 2, "/": 2, "^": 2, "exp": 5, "log": 5, "tanh": 17}


def to_program(expr: sp.Expr) -> tuple[str, list]:
    """A sympy expression in C, G as a program: every float becomes a parameter p[i]."""
    floats = sorted({a for a in expr.atoms(sp.Float)}, key=lambda f: str(f))
    ps = sp.symbols(f"p_0:{len(floats)}")
    expr = expr.xreplace(dict(zip(floats, ps)))
    body = sp.printing.numpy.NumPyPrinter().doprint(expr)
    body = re.sub(r"\bnumpy\.", "np.", body)
    body = re.sub(r"\bp_(\d+)\b", r"p[\1]", body)
    if not any(s.name in ("C", "G") for s in expr.free_symbols):  # a constant: still an array
        body = f"{body} + 0.0 * C"
    init = [float(f) for f in floats]
    return f"N_PARAMS = {len(init)}\nPARAM_INIT = {init!r}\n\ndef model(C, G, p):\n    return {body}\n", init


def predict(code: str, rows, p):
    ns = {"np": np, "math": math}
    exec(code, ns)
    with np.errstate(all="ignore"):
        out = ns["model"](rows["C"].to_numpy(float), rows["G"].to_numpy(float), np.asarray(p, float))
    return np.broadcast_to(np.asarray(out, float), (len(rows),))


def main():
    target, scale_name, split = sys.argv[1], sys.argv[2], int(sys.argv[3])
    data_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else HERE / "data"
    scale = float(scale_name.replace("_", "."))
    rows = pd.read_csv(data_dir / f"bins_r{split}.csv")
    sets = {s: rows[rows["set"] == s].reset_index(drop=True) for s in ("train", "val")}
    tr = sets["train"]
    out = HERE / "runs" / f"{target}_x{scale_name}_r{split}"
    out.mkdir(parents=True, exist_ok=True)

    from pysr import PySRRegressor                                 # starts Julia: slow, so imported here
    y = tr[target].to_numpy(float)
    baseline = pysr_loss(y, float(np.mean(y)), target)            # PySR's baseline: the loss of the mean
    nested = {u: {v: 0 for v in UNARY} for u in UNARY}              # no exp(log(..)), tanh(exp(..)), ...
    nested["^"] = {"^": 0, **{u: 0 for u in UNARY}}
    reg = PySRRegressor(
        complexity_of_variables=2, complexity_of_constants=5 + PARAM_WEIGHT / COMPLEXITY_WEIGHT,
        complexity_of_operators=COSTS, parsimony=scale * COMPLEXITY_WEIGHT * math.log(2) / baseline,
        binary_operators=BINARY, unary_operators=UNARY,
        constraints={"^": (-1, 1)}, nested_constraints=nested,          # exponents: a constant or a variable
        elementwise_loss=JULIA_LOSS[target],
        maxsize=250, niterations=100, populations=48, population_size=50, precision=64, random_state=0,
        model_selection="accuracy", progress=False, verbosity=1, output_directory=str(out), run_id="pysr",
    )
    reg.fit(tr[["C", "G"]].to_numpy(float), y, variable_names=["C", "G"])

    hof = []
    for i, r in reg.equations_.reset_index(drop=True).iterrows():
        expr = reg.sympy(i)
        code, init = to_program(expr)
        nodes = int(check_code(code, ["C", "G"], max_params=99, op_costs=expansion_costs())["complexity"])
        g = {s: gain_bits(sets[s][target].to_numpy(), predict(code, sets[s], init), target) for s in sets}
        hof.append({"pysr_complexity": int(r["complexity"]), "pysr_loss": float(r["loss"]), "equation": str(expr),
                    "n_params": len(init), "complexity": nodes, "train_gain_bits": g["train"],
                    "val_gain_bits": g["val"],
                    "score": g["val"] - scale * (PARAM_WEIGHT * len(init) + COMPLEXITY_WEIGHT * nodes),
                    "params": init, "code": code})
    hof = pd.DataFrame(hof)
    hof.drop(columns="code").to_csv(out / "hall_of_fame.csv", index=False)
    best = hof.loc[hof["score"].idxmax()]
    (out / "best.py").write_text(
        f"# PySR, {target}, penalty x{scale:g}, split {split}: the best hall-of-fame program by val score\n"
        f"# score {best['score']:.6f} (val gain {best['val_gain_bits']:.6f} minus penalty), "
        f"train gain {best['train_gain_bits']:.6f} bits/bin\n"
        f"# {best['n_params']} parameters, {best['complexity']} syntax-tree nodes\n"
        f"# equation (constants fitted on train): {target} = {best['equation']}\n"
        f"import math\n\nimport numpy as np\n\nFITTED_PARAMS = {list(map(float, best['params']))!r}\n\n{best['code']}")
    print(f"answer: {target} = {best['equation']}  (score {best['score']:.6f})")


if __name__ == "__main__":
    main()
