"""Turn a candidate program into a plain-text equation a person can read.

    mu    = 0
    kappa = p0 · rel/(rel + p1)/(1 + c/p2)
      where c   = max(C, 1e-6)
            rel = max(G, 0)/(c + p3)

Rules: temporaries used once and short enough are substituted into the line that
uses them; the rest are listed under "where". `p[i]` is written `p<i>` (or the fitted
value, with numeric=True). NumPy calls get their mathematical names (np.log -> ln,
np.maximum -> max, np.abs -> |x|, np.zeros_like(C) -> 0, ...). Programs the renderer
can't express (loops, branches) fall back to their code, so this never fails.
"""
from __future__ import annotations

import ast
import copy
import re

INLINE_MAX = 40            # inline single-use temporaries up to this many characters

FUNCS = {  # numpy / math name -> written name
    "maximum": "max", "fmax": "max", "minimum": "min", "fmin": "min", "exp": "exp",
    "log": "ln", "log10": "log10", "log2": "log2", "sqrt": "sqrt", "cbrt": "cbrt",
    "tanh": "tanh", "sinh": "sinh", "cosh": "cosh", "sin": "sin", "cos": "cos", "tan": "tan",
    "arctan": "atan", "arcsin": "asin", "arccos": "acos", "arctan2": "atan2", "atan2": "atan2",
    "arctanh": "atanh", "sign": "sign", "clip": "clip", "erf": "erf", "floor": "floor",
    "ceil": "ceil", "max": "max", "min": "min",
}
CONSTS = {"pi": "π", "e": "e", "inf": "∞"}
# precedence: higher binds tighter
P_ATOM, P_POW, P_UNARY, P_MUL, P_ADD, P_CMP, P_NOT, P_AND, P_OR, P_IF = 100, 80, 70, 60, 50, 40, 30, 20, 10, 5
BINOPS = {ast.Add: (" + ", P_ADD), ast.Sub: (" − ", P_ADD), ast.Mult: (" · ", P_MUL),
          ast.Div: ("/", P_MUL), ast.FloorDiv: (" // ", P_MUL), ast.Mod: (" mod ", P_MUL),
          ast.Pow: ("^", P_POW), ast.BitAnd: (" and ", P_AND), ast.BitOr: (" or ", P_OR)}
CMPOPS = {ast.Gt: " > ", ast.GtE: " ≥ ", ast.Lt: " < ", ast.LtE: " ≤ ", ast.Eq: " = ", ast.NotEq: " ≠ "}
MODULES = {"np", "numpy", "math"}


class Unsupported(Exception):
    pass


def _num(v: float) -> str:
    if isinstance(v, bool):
        return str(v)
    if float(v).is_integer() and abs(v) < 1e6:
        return str(int(v))
    s = f"{v:.4g}"
    return re.sub(r"e([+-])0*(\d)", lambda m: f"e{'' if m.group(1) == '+' else '-'}{m.group(2)}", s)


class Renderer:
    def __init__(self, params=None):
        self.params = params                      # numbers to substitute for p[i], or None

    def __call__(self, n) -> str:
        return self.r(n)[0]

    def wrap(self, n, prec, right=False, parent=None):
        s, p = self.r(n)
        need = p < prec
        if p == prec and right and parent is not None:
            # a − (b ± c) and a/(b·c) need brackets; a + (b ± c) and a·(b/c) don't
            need = isinstance(parent, (ast.Sub, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv))
        if p == prec and not right and isinstance(parent, ast.Pow):
            need = True
        return f"({s})" if need else s

    def r(self, n):  # -> (text, precedence)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                s = _num(n.value)
                return (s, P_ATOM) if not s.startswith("-") else (s, P_UNARY)
            return repr(n.value), P_ATOM
        if isinstance(n, ast.Name):
            return n.id, P_ATOM
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "p":
            i = n.slice
            if isinstance(i, ast.Constant) and isinstance(i.value, int):
                if self.params is not None and i.value < len(self.params):
                    s = _num(float(self.params[i.value]))
                    return (s, P_ATOM) if not s.startswith("-") else (f"({s})", P_ATOM)
                return f"p{i.value}", P_ATOM
            return f"p[{self(i)}]", P_ATOM
        if isinstance(n, ast.Subscript):
            return f"{self.wrap(n.value, P_ATOM)}[{self(n.slice)}]", P_ATOM
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in MODULES:
            if n.attr in CONSTS:
                return CONSTS[n.attr], P_ATOM
            raise Unsupported(f"attribute {n.attr}")
        if isinstance(n, ast.BinOp):
            op, prec = BINOPS[type(n.op)]
            if isinstance(n.op, ast.Pow):        # right-associative
                return f"{self.wrap(n.left, prec, False, n.op)}^{self.wrap(n.right, prec + 1)}", prec
            neg = self.negative(n.right)
            if neg is not None and isinstance(n.op, (ast.Add, ast.Sub)):
                op = " − " if isinstance(n.op, ast.Add) else " + "
                return f"{self.wrap(n.left, prec, False, n.op)}{op}{neg}", prec
            return (f"{self.wrap(n.left, prec, False, n.op)}{op}"
                    f"{self.wrap(n.right, prec, True, n.op)}"), prec
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                return f"−{self.wrap(n.operand, P_UNARY)}", P_UNARY
            if isinstance(n.op, ast.UAdd):
                return self.r(n.operand)
            if isinstance(n.op, ast.Not):
                return f"not {self.wrap(n.operand, P_NOT)}", P_NOT
        if isinstance(n, ast.Compare):
            parts = [self.wrap(n.left, P_CMP + 1)]
            for op, c in zip(n.ops, n.comparators):
                parts += [CMPOPS.get(type(op), " ? "), self.wrap(c, P_CMP + 1)]
            return "".join(parts), P_CMP
        if isinstance(n, ast.BoolOp):
            word, prec = (" and ", P_AND) if isinstance(n.op, ast.And) else (" or ", P_OR)
            return word.join(self.wrap(v, prec + 1) for v in n.values), prec
        if isinstance(n, ast.IfExp):
            return (f"{self.wrap(n.body, P_IF + 1)} if {self.wrap(n.test, P_IF + 1)} "
                    f"else {self.wrap(n.orelse, P_IF)}"), P_IF
        if isinstance(n, ast.Call):
            return self.call(n)
        if isinstance(n, (ast.Tuple, ast.List)):
            return "(" + ", ".join(self(e) for e in n.elts) + ")", P_ATOM
        raise Unsupported(type(n).__name__)

    def negative(self, n):
        """'x' if n is the number −x (a constant or a substituted parameter), else None."""
        v = None
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            v = n.value
        elif (self.params is not None and isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
              and n.value.id == "p" and isinstance(n.slice, ast.Constant)
              and isinstance(n.slice.value, int) and n.slice.value < len(self.params)):
            v = float(self.params[n.slice.value])
        return _num(-v) if v is not None and v < 0 else None

    def call(self, n):
        f = n.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in MODULES:
            name = f.attr
        elif isinstance(f, ast.Name):
            name = f.id
        else:
            raise Unsupported("call")
        a = n.args
        if name in ("zeros_like",):
            return "0", P_ATOM
        if name in ("ones_like",):
            return "1", P_ATOM
        if name == "full_like" and len(a) >= 2:
            return self.r(a[1])
        if name in ("asarray", "array", "float", "float64", "atleast_1d") and len(a) == 1:
            return self.r(a[0])
        if name in ("abs", "fabs", "absolute") and len(a) == 1:
            return f"|{self(a[0])}|", P_ATOM
        if name == "square" and len(a) == 1:
            return f"{self.wrap(a[0], P_POW + 1)}^2", P_POW
        if name in ("power", "pow") and len(a) == 2:
            return f"{self.wrap(a[0], P_POW + 1)}^{self.wrap(a[1], P_POW + 1)}", P_POW
        if name == "log1p" and len(a) == 1:
            return f"ln(1 + {self(a[0])})", P_ATOM
        if name == "expm1" and len(a) == 1:
            return f"exp({self(a[0])}) − 1", P_ADD
        if name == "hypot" and len(a) == 2:
            return f"sqrt({self.wrap(a[0], P_POW + 1)}^2 + {self.wrap(a[1], P_POW + 1)}^2)", P_ATOM
        if name == "where" and len(a) == 3:
            return (f"({self(a[1])} if {self(a[0])} else {self(a[2])})"), P_ATOM
        written = FUNCS.get(name, name)
        return f"{written}({', '.join(self(x) for x in a)})", P_ATOM


# ------------------------------------------------------------------ substitution

class _Subst(ast.NodeTransformer):
    def __init__(self, env):
        self.env = env

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Load) and n.id in self.env:
            return copy.deepcopy(self.env[n.id])
        return n


def _uses(node, name) -> int:
    return sum(1 for x in ast.walk(node) if isinstance(x, ast.Name) and x.id == name
               and isinstance(x.ctx, ast.Load))


def _p_index(n):
    return ast.Subscript(value=ast.Name("p", ast.Load()), slice=ast.Constant(n), ctx=ast.Load())


def _straight_line(body, params_name, inputs=()):
    """[(name, expr)] assignments and the return expression of a straight-line body.
    A name assigned again (an input such as C, or an earlier temporary) gets a prime,
    C -> C', so that every name in the equation has a single meaning."""
    raw, ret = [], None
    for st in body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue                                  # docstring
        if isinstance(st, ast.Assign) and len(st.targets) == 1:
            t, v = st.targets[0], st.value
            if isinstance(t, ast.Tuple) and all(isinstance(e, ast.Name) for e in t.elts):
                if isinstance(v, ast.Name) and v.id == params_name:       # a, b = p
                    for i, e in enumerate(t.elts):
                        raw.append((e.id, _p_index(i)))
                    continue
                if isinstance(v, (ast.Tuple, ast.List)) and len(v.elts) == len(t.elts):
                    for e, x in zip(t.elts, v.elts):
                        raw.append((e.id, x))
                    continue
            if isinstance(t, ast.Name):
                raw.append((t.id, v)); continue
        if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
            raw.append((st.target.id, ast.BinOp(ast.Name(st.target.id, ast.Load()), st.op, st.value)))
            continue
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value:
            raw.append((st.target.id, st.value)); continue
        if isinstance(st, ast.Return):
            ret = st.value; break
        raise Unsupported(type(st).__name__)
    if ret is None:
        raise Unsupported("no return")
    cur = {params_name: "p"}                          # name in the code -> name in the equation
    defined = set(inputs) | {"p"}

    def ren(x):
        return _Subst({k: ast.Name(v, ast.Load()) for k, v in cur.items() if k != v}).visit(copy.deepcopy(x))
    steps = []
    for k, v in raw:
        v = ren(v)
        new = k
        while new in defined:
            new += "'"
        defined.add(new); cur[k] = new
        steps.append((new, v))
    return steps, ren(ret)


def _simplify(steps, outputs, render):
    """Substitute away trivial and single-use temporaries; return kept steps and outputs."""
    names = [k for k, _ in steps]
    reassigned = {k for k in names if names.count(k) > 1}
    kept = []
    for i, (k, v) in enumerate(steps):
        later = [x for _, x in steps[i + 1:]] + list(outputs)
        n_uses = sum(_uses(x, k) for x in later)
        if n_uses == 0 and k not in reassigned:
            continue                                      # dead
        trivial = isinstance(v, (ast.Name, ast.Constant)) or (
            isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) and v.value.id == "p")
        in_outputs_only = n_uses == 1 and not any(_uses(x, k) for _, x in steps[i + 1:])
        short = n_uses == 1 and (len(render(v)) <= INLINE_MAX or in_outputs_only)
        if k not in reassigned and (trivial or short):
            sub = _Subst({k: v})
            steps = steps[:i + 1] + [(kk, sub.visit(vv)) for kk, vv in steps[i + 1:]]
            outputs = [sub.visit(o) for o in outputs]
            continue
        kept.append((k, v))
    # keep only the definitions the outputs still need
    need, final = set(), []
    for k, v in reversed(kept):
        if any(_uses(o, k) for o in outputs) or k in need:
            final.append((k, v))
            need |= {x.id for x in ast.walk(v) if isinstance(x, ast.Name)}
    return list(reversed(final)), outputs


def to_lines(code: str, params=None, outputs=("mu", "kappa")) -> list[str]:
    """The equation as lines of text. With params, p[i] are replaced by their values."""
    tree = ast.parse(code)
    render = Renderer(params)
    model = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "model")
    consts = {}
    for st in tree.body:                                   # module-level constants
        if (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                and st.targets[0].id not in ("N_PARAMS", "PARAM_INIT")):
            consts[st.targets[0].id] = st.value
    pname = model.args.args[-1].arg
    steps, ret = _straight_line(model.body, pname, [a.arg for a in model.args.args[:-1]])
    if consts:
        sub = _Subst(consts)
        steps = [(k, sub.visit(v)) for k, v in steps]
        ret = sub.visit(ret)
    outs = list(ret.elts) if isinstance(ret, ast.Tuple) and len(ret.elts) == len(outputs) else None
    if outs is None and len(outputs) == 1 and not isinstance(ret, (ast.Tuple, ast.List)):
        outs = [ret]                                       # a single output needs no tuple
    if outs is None:
        raise Unsupported(f"model must return ({', '.join(outputs)})")
    steps, outs = _simplify(steps, outs, render)
    width = max(len(o) for o in outputs)
    lines = [f"{o.ljust(width)} = {render(e)}" for o, e in zip(outputs, outs)]
    if steps:
        w = max(len(k) for k, _ in steps)
        for j, (k, v) in enumerate(steps):
            lines.append(f"  {'where' if j == 0 else '     '} {k.ljust(w)} = {render(v)}")
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "model"]
    for h in helpers:
        hargs = [a.arg for a in h.args.args]
        hs, hr = _straight_line(h.body, "p", [x for x in hargs if x != "p"])
        hs, hr = _simplify(hs, [hr], render)
        args = ", ".join(a.arg for a in h.args.args)
        lines.append(f"  {h.name}({args}) = {render(hr[0])}"
                     + "".join(f";  {k} = {render(v)}" for k, v in hs))
    return lines


def equation(code: str, params=None, numeric: bool = False, outputs=("mu", "kappa")) -> str:
    """Multi-line equation. Never raises: falls back to the model's code."""
    try:
        lines = to_lines(code, params if numeric else None, outputs)
        if params is not None and not numeric:
            lines.append("  with  " + ", ".join(f"p{i} = {_num(float(v))}" for i, v in enumerate(params)))
        return "\n".join(lines)
    except Exception as e:                              # anything the renderer can't express
        body = code.split("def model", 1)[-1]
        return f"(no closed form: {type(e).__name__}: {e})\ndef model{body}"


def one_line(code: str, outputs=("mu", "kappa")) -> str:
    """For the CSV: the same equation on one line, statements separated by '; '."""
    try:
        return "; ".join(" ".join(l.split()) for l in to_lines(code, None, outputs)).replace("; where ", "; ")
    except Exception:
        return ""
