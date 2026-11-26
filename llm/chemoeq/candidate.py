"""Parse, check, fit and score one LLM-proposed model.

A candidate is a small Python program whose arguments are the experiment's inputs:

    N_PARAMS = 2
    PARAM_INIT = [1.0, 100.0]          # optional starting guesses
    def model(C, G, p):                # e.g. inputs {"C": "C", "G": "G"}
        ...
        return mu, kappa               # what the experiment's likelihood expects

Parameters are fitted by maximum likelihood on the training conditions; the score
is the information gained on the validation conditions over a uniformly random
direction, in bits per segment, minus a complexity penalty.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import math
import multiprocessing as mp
import os
import re
import time
import traceback
import types

import numpy as np
from scipy.optimize import minimize

from .likelihood import gain_bits

PARAM_CAP = 1e9
# Rows per piece when a fit evaluates the likelihood: pieces this small keep the arrays and the
# program's intermediate results in the CPU cache. On the whole table (~300,000 rows) they spill
# to main memory, which every core shares: a fit is then ~3x slower alone, and several fits at
# once mostly wait on memory (measured on a 12-core Ryzen: 1.35 fits/s at best with whole arrays,
# 5.8 with pieces and 22 fits at once).
CHUNK_ROWS = 16384

# ---------------------------------------------------------------- code rules
#
# LLM-written programs are checked statically, then run with a reduced set of
# builtins, in worker processes that cannot write files and have a memory cap
# (see _limit_worker). Together these stop the known ways out of the checks
# (reaching `builtins`, `importlib` or frame objects through numpy's own
# submodules or through generators). This is defence in depth, not an OS-level
# sandbox: for full isolation run the search in a container without network
# access (see DOCUMENTATION.md, "Security").
#
# The rules:
#   - imports: only `import numpy [as np]`, `import math`, and
#     `from numpy|math import <function>`;
#   - np / numpy / math may only be used as `np.<name>` (or `np.linalg.<name>`),
#     never passed around, assigned or rebound, and `<name>` must not be a module;
#   - no attribute or name that starts with "_", no class definitions,
#     no file-, OS- or introspection-related attribute names.

MODULES = {"np": np, "numpy": np, "math": math}
SAFE_SUBMODULES = {"linalg"}                 # np.linalg.<fn> is allowed
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "globals", "locals",
                   "getattr", "setattr", "delattr", "input", "breakpoint", "vars", "dir",
                   "type", "object", "memoryview", "exit", "quit", "help", "super",
                   "classmethod", "staticmethod", "property"}
# names of numpy/ndarray attributes that touch files, the OS or native code,
# and of Python's frame/code introspection attributes
FORBIDDEN_ATTRS = {"os", "sys", "subprocess", "ctypeslib", "ctypes", "lib", "testing", "f2py",
                   "distutils", "load", "loadtxt", "genfromtxt", "fromfile", "tofile", "save",
                   "savez", "savez_compressed", "savetxt", "memmap", "fromregex", "DataSource",
                   "show_config", "show_runtime", "system", "popen", "dump", "dumps",
                   "builtins", "importlib", "inspect", "mro", "format_map"}
FORBIDDEN_ATTR_PREFIXES = ("_", "gi_", "cr_", "ag_", "f_", "tb_", "co_", "func_")

# The only builtins a candidate program can see.
_SAFE_BUILTIN_NAMES = (
    "abs all any bool complex dict divmod enumerate filter float frozenset int isinstance "
    "len list map max min pow print range reversed round set slice sorted sum tuple zip "
    "ArithmeticError Exception FloatingPointError IndexError KeyError OverflowError "
    "RuntimeError TypeError ValueError ZeroDivisionError").split()


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name not in ("numpy", "math"):
        raise ImportError(f"import of {name!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES}
SAFE_BUILTINS["__import__"] = _guarded_import

# ---------------------------------------------------------------- operator costs
#
# The plain complexity counts syntax-tree nodes, so np.tanh(x) costs what np.exp(x) or x / (1 + x) costs.
# With operator costs (Experiment.op_costs), a function counts as what it takes to write it with arithmetic,
# np.exp and np.log: each use of np.<f> / math.<f> adds the nodes of its expansion below (argument x counted
# once) minus those of np.<f>(x). Functions with no such expansion (trigonometric, erf, gamma, ...) cost the
# most expensive expansion, so none of them is a cheaper way to a sigmoid than tanh. Anything else, np.exp and
# np.log included, costs its nodes as before.
EXPANSIONS = {
    "tanh": "1 - 2 / (1 + np.exp(2 * x))",
    "sinh": "(np.exp(x) - np.exp(-x)) / 2",
    "cosh": "(np.exp(x) + np.exp(-x)) / 2",
    "arcsinh": "np.log(x + (x * x + 1) ** 0.5)",
    "arccosh": "np.log(x + (x * x - 1) ** 0.5)",
    "arctanh": "np.log((1 + x) / (1 - x)) / 2",
    "expm1": "np.exp(x) - 1",
    "log1p": "np.log(1 + x)",
    "exp2": "np.exp(x * 0.6931471805599453)",
    "log2": "np.log(x) / 0.6931471805599453",
    "log10": "np.log(x) / 2.302585092994046",
    "logaddexp": "np.log(np.exp(x) + np.exp(y))",
    "hypot": "(x * x + y * y) ** 0.5",
    "deg2rad": "x * 0.017453292519943295", "radians": "x * 0.017453292519943295",
    "rad2deg": "x * 57.29577951308232", "degrees": "x * 57.29577951308232",
}
NO_EXPANSION = "sin cos tan arcsin arccos arctan arctan2 asin acos atan atan2 sinc erf erfc gamma lgamma i0".split()


def _nodes(src: str) -> int:
    return sum(1 for _ in ast.walk(ast.parse(src, mode="eval").body))


def expansion_costs() -> dict:
    """Extra nodes per use of each function: its expansion minus np.<f>(x) (see EXPANSIONS)."""
    out = {f: max(0, _nodes(e) - _nodes(f"np.{f}(x, y)" if "y" in e else f"np.{f}(x)"))
           for f, e in EXPANSIONS.items()}
    top = max(out.values())
    return {**out, **{f: top for f in NO_EXPANSION}}


def _op_surcharge(tree, body, op_costs: dict) -> int:
    """Extra nodes for every use, in `body`, of a costed function: np.f / math.f (called or not, so an alias
    pays too), and names imported anywhere in `tree` with `from numpy|math import f`."""
    imported = {(a.asname or a.name): a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names}
    extra = 0
    for n in (m for b in body for m in ast.walk(b)):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in MODULES:
            extra += op_costs.get(n.attr, 0)
        elif isinstance(n, ast.Name) and n.id in imported:
            extra += op_costs.get(imported[n.id], 0)
    return extra

class CandidateError(Exception):
    pass


# ---------------------------------------------------------------- parsing

def extract_code(text: str) -> str:
    """Take the last ```python block that defines model(); fall back to raw text."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    for b in reversed(blocks):
        if "def model" in b:
            return b.strip()
    if "def model" in text:
        return text[text.index("N_PARAMS") if "N_PARAMS" in text else text.index("def model"):].strip()
    raise CandidateError("no `def model` found in response")


def extract_idea(text: str, limit: int = 300) -> str:
    """The model's one- or two-line idea: the text before its first code block."""
    head = text.split("```", 1)[0] if "```" in text else text
    head = " ".join(head.split())
    return head[:limit] + ("..." if len(head) > limit else "")


def check_code(code: str, args, max_params: int, op_costs: dict | None = None) -> dict:
    """Static checks; returns metadata (n_params, init, complexity, canonical hash).
    `args` are the program's input names; the signature must be model(*args, p).
    `op_costs`: extra nodes per use of a function (see expansion_costs), added to the complexity."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CandidateError(f"syntax error: {e}") from None
    model_fn = None
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ("numpy", "math") or MODULES.get(a.asname or a.name) is not MODULES[a.name]:
                    raise CandidateError(f"import not allowed: {a.name}"
                                         + (f" as {a.asname}" if a.asname else "")
                                         + " (use `import numpy as np` or `import math`)")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ("numpy", "math"):
                raise CandidateError(f"import not allowed: from {node.module}")
            for a in node.names:
                _check_module_attr(MODULES[node.module], a.name, final=True)
                if a.asname and (a.asname in MODULES or a.asname.startswith("_")):
                    raise CandidateError(f"import alias not allowed: {a.asname}")
        elif isinstance(node, ast.ExceptHandler) and node.name in MODULES:
            raise CandidateError(f"name not allowed: {node.name}")
        elif isinstance(node, ast.Match):
            raise CandidateError("match statements not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise CandidateError(f"name not allowed: {node.id}")
            if node.id in MODULES:
                _check_module_use(node, parents)
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS or node.attr.startswith(FORBIDDEN_ATTR_PREFIXES):
                raise CandidateError(f"attribute not allowed: {node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise CandidateError("global/nonlocal not allowed")
        elif isinstance(node, ast.ClassDef):
            raise CandidateError("class definitions not allowed")
        elif isinstance(node, ast.arg) and (node.arg in MODULES or node.arg.startswith("__")):
            raise CandidateError(f"argument name not allowed: {node.arg}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in MODULES:
            raise CandidateError(f"function name not allowed: {node.name}")
        if isinstance(node, ast.FunctionDef) and node.name == "model":
            model_fn = node
    if model_fn is None:
        raise CandidateError("no function named model")
    want = list(args) + ["p"]
    if [a.arg for a in model_fn.args.args] != want:
        raise CandidateError(f"model signature must be model({', '.join(want)}), "
                             f"got model({', '.join(a.arg for a in model_fn.args.args)})")
    ns = _exec(code)
    n_params = ns.get("N_PARAMS")
    if not isinstance(n_params, int) or not (0 <= n_params <= max_params):
        raise CandidateError(f"N_PARAMS must be an int in 0..{max_params}")
    init = ns.get("PARAM_INIT", [1.0] * n_params)
    init = np.asarray(init, float).ravel()
    if init.size != n_params or not np.all(np.isfinite(init)):
        raise CandidateError("PARAM_INIT must be a list of N_PARAMS finite numbers")
    # Complexity and identity cover the whole program (helper functions and
    # module-level constants included), minus the N_PARAMS / PARAM_INIT lines.
    # Counting only model() would let logic hide in helpers, escape the
    # complexity penalty, and make different programs look like duplicates.
    body = [n for n in tree.body if not _is_meta_assign(n)]
    complexity = sum(1 for n in body for _ in ast.walk(n))
    if op_costs:
        complexity += _op_surcharge(tree, body, op_costs)
    canon = hashlib.sha1("".join(ast.dump(n, annotate_fields=False) for n in body).encode()).hexdigest()[:12]
    return {"n_params": n_params, "init": init, "complexity": complexity, "hash": canon}


def _check_module_attr(obj, attr: str, final: bool):
    """One step of `np.a.b`: returns getattr(obj, attr) if it is allowed there."""
    if attr in FORBIDDEN_ATTRS or attr.startswith(FORBIDDEN_ATTR_PREFIXES):
        raise CandidateError(f"attribute not allowed: {attr}")
    if not hasattr(obj, attr):
        raise CandidateError(f"{getattr(obj, '__name__', obj)} has no attribute {attr!r}")
    val = getattr(obj, attr)
    if isinstance(val, types.ModuleType) and (final or attr not in SAFE_SUBMODULES):
        raise CandidateError(f"module access not allowed: {attr} (use functions such as np.exp)")
    return val


def _check_module_use(name_node, parents):
    """np / numpy / math may only appear as the root of `np.f` or `np.linalg.f`."""
    obj, node, chain = MODULES[name_node.id], name_node, []
    if not isinstance(node.ctx, ast.Load):
        raise CandidateError(f"`{name_node.id}` cannot be assigned or deleted")
    while isinstance(parents.get(node), ast.Attribute) and parents[node].value is node:
        node = parents[node]
        chain.append(node.attr)
    if not chain:
        raise CandidateError(f"`{name_node.id}` may only be used as {name_node.id}.<function>")
    for i, attr in enumerate(chain):
        obj = _check_module_attr(obj, attr, final=(i == len(chain) - 1))
        if not isinstance(obj, types.ModuleType):
            break       # e.g. np.add.reduce: later steps are checked as plain attributes


def _is_meta_assign(node) -> bool:
    return (isinstance(node, (ast.Assign, ast.AnnAssign)) and
            any(isinstance(t, ast.Name) and t.id in ("N_PARAMS", "PARAM_INIT")
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target])))


def _exec(code: str) -> dict:
    ns = {"np": np, "numpy": np, "math": math, "__builtins__": SAFE_BUILTINS}
    exec(compile(code, "<candidate>", "exec"), ns)  # noqa: S102 - code passed check_code
    return ns


# ---------------------------------------------------------------- scoring

class Model:
    """A checked program bound to its input names and the experiment's likelihood."""

    def __init__(self, code: str, args, likelihood):
        self.fn = _exec(code)["model"]
        self.args, self.lik = tuple(args), likelihood
        self.chunked = None             # nll in pieces of CHUNK_ROWS: None until checked on the first call

    def _rows(self, X) -> int:
        return X[self.args[0]].shape[0] if self.args else X["theta"].shape[0]

    @staticmethod
    def _pieces(X, n):
        """X cut into pieces of CHUNK_ROWS rows (arrays with one value per row; others as they are)."""
        for a in range(0, n, CHUNK_ROWS):
            yield {k: v[a:a + CHUNK_ROWS] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n else v
                   for k, v in X.items()}

    def _same_in_pieces(self, X, p, n) -> bool:
        """Whether the program gives every row the same values in pieces as on the whole table. Not
        so for one that uses statistics of all rows (np.mean(C), np.max(G), ...): it is then
        always evaluated whole. Checked at p and at a nearby point."""
        try:
            for q in (p, np.asarray(p, float) * 1.1 + 0.01):
                whole = self.outputs(X, q)
                parts = [self.outputs(Xi, q) for Xi in self._pieces(X, n)]
                for j, w in enumerate(whole):
                    if not np.allclose(np.concatenate([o[j] for o in parts]), w, rtol=1e-12, atol=0, equal_nan=True):
                        return False
        except Exception:
            return False
        return True

    def outputs(self, X, p):
        """The likelihood's parameters for every row of X (broadcast and cleaned)."""
        with np.errstate(all="ignore"):
            out = self.fn(*(X[a] for a in self.args), p)
        n = X[self.args[0]].shape[0] if self.args else X["theta"].shape[0]
        if len(self.lik.outputs) == 1 and not isinstance(out, (tuple, list)):
            out = (out,)                                   # a single output needs no tuple
        if not isinstance(out, (tuple, list)) or len(out) != len(self.lik.outputs):
            raise CandidateError(f"model must return {len(self.lik.outputs)} values "
                                 f"({', '.join(self.lik.outputs)})")
        if hasattr(self.lik, "prepare_with"):              # predictions per class of previous values, averaged
            return self.lik.prepare_with(out, X, n)
        return self.lik.prepare(out, n)

    def logpdf(self, X, p):
        return self.lik.logpdf(self.lik.target(X), *self.outputs(X, p))

    def mean_cos(self, X, p):
        return self.lik.mean_cos(*self.outputs(X, p))

    def nll(self, X, p) -> float:
        """Mean negative log-likelihood per segment over the rows of X (weighted by X["weight"] for
        bin rows); inf if any row's is not finite. Evaluated in pieces of CHUNK_ROWS rows when the
        program allows it (see CHUNK_ROWS)."""
        n = self._rows(X)
        per = float(np.sum(X["weight"])) if "weight" in X else n
        if n > CHUNK_ROWS and self.chunked is None:
            self.chunked = self._same_in_pieces(X, p, n)
        if n <= CHUNK_ROWS or not self.chunked:
            return self._nll_sum(X, p) / per
        total = 0.0
        for Xi in self._pieces(X, n):
            total += self._nll_sum(Xi, p)
            if not np.isfinite(total):
                return np.inf
        return total / per

    def _nll_sum(self, X, p) -> float:
        try:
            out = self.outputs(X, p)
        except Exception:
            return np.inf
        if not all(np.all(np.isfinite(v)) for v in out):
            return np.inf
        with np.errstate(all="ignore"):
            lp = self.lik.logpdf(self.lik.target(X), *out)
            if "weight" in X:                           # bin rows: weight x the log density of the bin's mean
                lp = lp * X["weight"]
        return float(-np.sum(lp)) if np.all(np.isfinite(lp)) else np.inf


def _objective(model: Model, X):
    """What the fits minimise: the mean negative log-likelihood, 1e6 where it is not
    finite or a parameter is beyond PARAM_CAP."""
    def obj(p):
        if np.any(np.abs(p) > PARAM_CAP):
            return 1e6
        v = model.nll(X, p)
        return v if np.isfinite(v) else 1e6
    return obj


def fit_starts(init, restarts, rng) -> list:
    """The starting points of a fit, in order: PARAM_INIT, then restarts - 1 spread
    log-uniformly over +-1.5 decades around it (signs kept)."""
    starts = [init]
    for _ in range(max(restarts - 1, 0)):
        base = np.where(init == 0, 1.0, init)
        starts.append(base * 10 ** rng.uniform(-1.5, 1.5, init.size))
    return starts


def fit_from(model: Model, X, start, maxiter=300):
    """One L-BFGS-B run from `start`: (parameters, value), or None if it failed."""
    try:
        r = minimize(_objective(model, X), start, method="L-BFGS-B", options={"maxiter": maxiter})
    except Exception:
        return None
    return np.asarray(r.x, float), float(r.fun)


def fit_finish(model: Model, X, init, runs):
    """The fit from the L-BFGS-B `runs` of its starts (in their order; None for a failed
    run): the best of them and of PARAM_INIT itself, then a Nelder-Mead polish."""
    if init.size == 0:
        return np.zeros(0), model.nll(X, np.zeros(0))
    obj = _objective(model, X)
    best_p, best_v = init, obj(init)
    for run in runs:
        if run is not None and run[1] < best_v - 1e-7:
            best_p, best_v = run
    return _polish(obj, best_p, best_v)


def _polish(obj, best_p, best_v):
    k = len(best_p)
    try:
        r = minimize(obj, best_p, method="Nelder-Mead",
                     options={"maxiter": 100 * k, "xatol": 1e-8, "fatol": 1e-10})
        if r.fun < best_v:
            best_p, best_v = r.x, r.fun
    except Exception:
        pass
    return np.asarray(best_p, float), float(best_v)


def fit(model: Model, X, init, restarts, rng, maxiter=300):
    """Multi-start L-BFGS-B, then Nelder-Mead polish. Scales unknown, so starts are
    spread log-uniformly over +-1.5 decades around PARAM_INIT (signs kept).
    Restarts stop early once two have converged to the same best value. (The
    Evaluator can instead run every start in a worker of its own and finish with
    fit_finish: the same fit without the early stop.)"""
    k = init.size
    if k == 0:
        return np.zeros(0), model.nll(X, np.zeros(0))
    obj = _objective(model, X)
    starts = fit_starts(init, restarts, rng)
    best_p, best_v, agree = init, obj(init), 0
    for s in starts:
        try:
            r = minimize(obj, s, method="L-BFGS-B", options={"maxiter": maxiter})
        except Exception:
            continue
        if r.fun < best_v - 1e-7:
            best_p, best_v, agree = r.x, r.fun, 1
        elif abs(r.fun - best_v) <= 1e-7:
            agree += 1
            if agree >= 2:          # two starts agree on the optimum: stop early
                break
    return _polish(obj, best_p, best_v)


def per_condition(model: Model, X, p):
    """Observed vs predicted mean cos(theta) and gain, per condition."""
    pred, lp, obs = model.mean_cos(X, p), model.logpdf(X, p), model.lik.observed(X)
    w = X["weight"] if "weight" in X else np.ones(len(lp))      # bin rows: weighted by their segments
    rows = []
    for c in np.unique(X["cond"]):
        m = X["cond"] == c
        rows.append({"cond": str(c), "n": int(round(w[m].sum())),
                     "obs_cos": float(np.average(obs[m], weights=w[m])),
                     "pred_cos": float(np.average(pred[m], weights=w[m])),
                     "gain_bits": float(gain_bits(np.average(lp[m], weights=w[m]), model.lik))})
    return rows


def fingerprint(model: Model, probe, p):
    """Behavioural fingerprint on a fixed probe set, to drop re-worded duplicates."""
    try:
        return hashlib.sha1(np.round(model.mean_cos(probe, p), 4).tobytes()).hexdigest()[:12]
    except Exception:
        return None


def evaluate(code, train, val, spec: dict, seed=0, probe=None, diagnostics=False, diag_data=None, runs=None):
    """Check, fit on train, score on val. Returns a result dict (never raises).
    spec: args, likelihood, max_params, fit_restarts, fit_maxiter, complexity_weight,
    param_weight (Experiment.scoring()). `runs`: the L-BFGS-B runs of the fit's starts,
    already done (in workers of their own, see Evaluator); the fit then only finishes."""
    res = {"code": code, "ok": False}
    try:
        meta = check_code(code, spec["args"], spec["max_params"], spec.get("op_costs"))
        model = Model(code, spec["args"], spec["likelihood"])
        if runs is None:
            p, train_nll = fit(model, train, meta["init"], spec["fit_restarts"],
                               np.random.default_rng(seed), spec["fit_maxiter"])
        else:
            p, train_nll = fit_finish(model, train, meta["init"], runs)
        val_nll = model.nll(val, p)
        if not np.isfinite(val_nll):
            raise CandidateError("non-finite likelihood on validation conditions")
        tg, vg = gain_bits(-train_nll, spec["likelihood"]), gain_bits(-val_nll, spec["likelihood"])
        score = vg - spec["complexity_weight"] * meta["complexity"] - spec["param_weight"] * meta["n_params"]
        res.update(ok=True, params=p.tolist(), n_params=meta["n_params"],
                   complexity=meta["complexity"], hash=meta["hash"],
                   train_gain=tg, val_gain=vg, score=score)
        if probe is not None:
            res["fingerprint"] = fingerprint(model, probe, p)
        if diagnostics:
            # one line per condition of the fitting rows, or per bin of `diag_data` (search.feedback_data)
            res["diagnostics"] = per_condition(model, train if diag_data is None else diag_data, p)
    except CandidateError as e:
        res["error"] = str(e)
    except MemoryError:
        res["error"] = "ran out of memory (per-worker limit: fit_memory_gb)"
    except Exception as e:  # LLM code can fail in any way
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc(limit=3)
    return res


# ---------------------------------------------------------------- worker processes

_SHARED = {}


def memory_limit_bytes(setting, n_procs: int) -> int | None:
    """Per-worker memory cap. "auto": 4 GB, or less if n_procs x 4 GB would
    exceed 75% of physical RAM. None / 0 disables the cap."""
    if not setting:
        return None
    if setting == "auto":
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            ram = 16 << 30
        return int(max(1 << 30, min(4 << 30, 0.75 * ram / max(n_procs, 1))))
    return int(float(setting) * (1 << 30))


def _limit_worker(mem_bytes: int | None):
    """Applied once in every worker before it runs LLM-written code:
      - no file writes at all (RLIMIT_FSIZE = 0; fit workers never need to write),
      - address space capped at what the worker already uses + mem_bytes,
      - no core dumps.
    Hard limits are lowered too, so the worker cannot raise them again.
    Linux only; elsewhere this does nothing."""
    try:
        import resource
    except ImportError:
        return
    # stdout/stderr redirected to a regular file (`> log`, nohup) would hit the
    # size limit on the first warning; send the worker's copies to /dev/null.
    import stat
    devnull = os.open(os.devnull, os.O_WRONLY)
    for fd in (1, 2):
        try:
            if stat.S_ISREG(os.fstat(fd).st_mode):
                os.dup2(devnull, fd)
        except OSError:
            pass
    os.close(devnull)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if mem_bytes and hasattr(resource, "RLIMIT_AS"):
        try:
            with open("/proc/self/statm") as f:
                used = int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        except OSError:
            used = 0
        cap = used + mem_bytes
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def _init_worker(shared, mem_bytes=None):
    _SHARED.clear()
    _SHARED.update(shared)
    _limit_worker(mem_bytes)


def _worker(args):
    code, seed = args[:2]
    runs = args[2] if len(args) > 2 else None
    s = _SHARED
    return evaluate(code, s["train"], s["val"], s["spec"], seed, s["probe"], s["diag"], s.get("diag_data"), runs)


def _start_worker(args):
    """One start of a candidate's fit (L-BFGS-B), in a worker of its own:
    {"run": (parameters, value) or None} or, if the program is rejected, {"error": ...}."""
    code, seed, i = args
    s = _SHARED
    try:
        meta = check_code(code, s["spec"]["args"], s["spec"]["max_params"], s["spec"].get("op_costs"))
        if meta["n_params"] == 0:
            return {"run": None}
        model = Model(code, s["spec"]["args"], s["spec"]["likelihood"])
        start = fit_starts(meta["init"], s["spec"]["fit_restarts"], np.random.default_rng(seed))[i]
        run = fit_from(model, s["train"], start, s["spec"]["fit_maxiter"])
        return {"run": None if run is None else (run[0].tolist(), run[1])}
    except CandidateError as e:
        return {"error": str(e)}
    except MemoryError:
        return {"error": "ran out of memory (per-worker limit: fit_memory_gb)"}
    except Exception as e:  # LLM code can fail in any way
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=3)}


def n_workers(setting) -> int:
    if setting in (None, "auto"):
        return max(1, min(32, (os.cpu_count() or 2) - 2))
    return int(setting)


class Evaluator:
    """Runs candidate fits in separate worker processes with a hard timeout.

    Workers start from a clean server process ("forkserver"), not by forking this
    one: this process also runs the threads that talk to the LLM, and forking a
    multi-threaded process can deadlock. Data are sent once per worker at start-up.
    A worker that exceeds the timeout is killed and the pool rebuilt. Workers
    cannot write files and have a memory cap (_limit_worker).

    With spec["parallel_restarts"] every start of every candidate's fit is a job of its
    own (fit_restarts jobs per candidate), so a batch of 8 candidates keeps up to
    8 x fit_restarts cores busy; a second job per candidate then picks the best start,
    polishes it and scores it (fit_finish). This is the same fit as `fit`, without its
    early stop once two starts agree. Each job gets the timeout; a candidate with a
    job that runs out of time fails as a whole, as a single fit would.
    """

    def __init__(self, spec, train, val, probe=None, diagnostics=True, timeout_s=None,
                 workers="auto", memory_gb="auto", diag_data=None):
        self.shared = dict(spec=spec, train=train, val=val, probe=probe, diag=diagnostics, diag_data=diag_data)
        self.timeout = timeout_s or spec["fit_timeout_s"]
        self.n = n_workers(workers)
        self.mem = memory_limit_bytes(memory_gb, self.n)
        self.ctx = mp.get_context("forkserver")
        self.pool = self._new_pool()

    def _new_pool(self):
        return self.ctx.Pool(self.n, initializer=_init_worker, initargs=(self.shared, self.mem))

    @property
    def restarts(self) -> int:
        """Jobs per candidate in the first phase: its fit's starts, if they run in parallel."""
        spec = self.shared["spec"]
        return int(spec["fit_restarts"]) if spec.get("parallel_restarts") and spec["fit_restarts"] > 1 else 0

    def submit(self, codes, seed=0):
        """Start fitting `codes` in the background; pass the handle to collect()."""
        codes, seeds = list(codes), [seed + i for i in range(len(codes))]
        if self.restarts:                       # first every start of every candidate
            jobs = [self.pool.apply_async(_start_worker, ((c, s, r),))
                    for c, s in zip(codes, seeds) for r in range(self.restarts)]
        else:
            jobs = [self.pool.apply_async(_worker, ((c, s),)) for c, s in zip(codes, seeds)]
        return time.time(), codes, seeds, jobs

    def _wait(self, start, jobs):
        """Each job's result, or None if it ran out of time. Each job gets `timeout`
        seconds of wall time from when it can start; queued jobs aren't charged for
        waiting. Only one batch is in flight at a time, so a pool rebuilt after a
        timeout loses no other work."""
        # a job can't start before ceil((i+1)/n) "slots" have passed, so allow for queueing
        out = []
        for i, j in enumerate(jobs):
            deadline = start + self.timeout * (i // self.n + 1)
            try:
                out.append(j.get(timeout=max(1.0, deadline - time.time())))
            except mp.TimeoutError:
                out.append(None)
        if any(o is None for o in out):
            self.pool.terminate()
            self.pool = self._new_pool()
        return out

    def collect(self, handle):
        """Results in the same order as the submitted codes."""
        start, codes, seeds, jobs = handle
        late = {"ok": False, "error": f"fit timed out (> {self.timeout:.0f}s)"}
        if not self.restarts:
            return [o if o is not None else {"code": c, **late} for c, o in zip(codes, self._wait(start, jobs))]
        got, R = self._wait(start, jobs), self.restarts
        out, finish = [None] * len(codes), []
        for i, c in enumerate(codes):
            parts = got[i * R:(i + 1) * R]
            if any(p is None for p in parts):
                out[i] = {"code": c, **late}
            elif any("error" in p for p in parts):     # rejected (the same in every start), or out of memory
                err = next(p for p in parts if "error" in p)
                out[i] = {"code": c, "ok": False, **err}
            else:                               # then pick the best start, polish and score
                runs = [None if p["run"] is None else (np.asarray(p["run"][0], float), p["run"][1]) for p in parts]
                finish.append((i, self.pool.apply_async(_worker, ((c, seeds[i], runs),))))
        for (i, _), o in zip(finish, self._wait(time.time(), [j for _, j in finish])):
            out[i] = o if o is not None else {"code": codes[i], **late}
        return out

    def run(self, codes, seed=0):
        """submit() and wait for the results."""
        return self.collect(self.submit(codes, seed))

    def close(self):
        self.pool.terminate()
