"""The paper's figure panels, from score.py's tables: one SVG per panel in results/figures/, named after the paper
(fig2A.svg, ..., S8C.svg).

    python analysis/figures.py [--results results]

  Fig. 2, 3   CEI, speed: A, C the held-back share of the best equation so far over the LLM and PySR searches (x1);
              B, D observed against predicted, every final equation on every bin; E, F the split 0 LLM and PySR
              equations over concentration and gradient, the LLM's in the parts it wrote it in
  Fig. 4      CEI: A, B held-back share against complexity at every penalty; C, D the split 0 equations at x10
  Fig. 5      CEI with other seed equations and prompts: A the held-back share of each final equation (triangle: it
              follows G/C); B each set's median equation
  S1-S2 Fig   CEI: every x1 final equation over concentration and gradient (LLM, PySR); S4-S5 cos theta, S6-S7 speed
  S3 Fig      cos theta: as Fig. 2A-D
  S8 Fig      speed: as Fig. 4
"""
import argparse, ast, json, math, re, textwrap
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSC, LogNorm, Normalize
from matplotlib.ticker import FuncFormatter

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e9e8e4", "#ffffff"
COL = {"llm": "#2a78d6", "pysr": "#eb6834"}
RAMP = {"llm": LSC.from_list("greenblue", ["#b9ecb0", "#3fc48a", "#1594b8", "#2a5fd6", "#0a1640"]),
        "pysr": LSC.from_list("yellowred", ["#f7e08a", "#f2b01e", "#ec6a2c", "#c4281f", "#4a0a0a"])}
HEAT, GREY = plt.cm.inferno, LSC.from_list("match", ["#ffffff", "#0b0b0b"])
NAME = {"cei": "CEI", "speed": "Speed", "costheta": "cos θ"}
UNIT = {"cei": "CEI", "speed": "speed (µm / min)", "costheta": "cos θ"}
METHOD = {"llm": "LLM", "pysr": "PySR"}
SCALES = [0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0]
SHARE = "solution fit on held-back data"
C_LABEL, G_LABEL = "concentration C (nM)", "gradient G (nM / µm)"
PENALTY = "complexity penalty of the solution at ×1 (mbit)"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none", "axes.labelcolor": INK,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.color": INK, "ytick.color": INK,
                     "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.dpi": 600})
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
POW10 = FuncFormatter(lambda v, _: "10" + str(int(round(np.log10(v)))).translate(SUP))
T = {}                                                             # the tables, read in main()


def figure(out, name, title, size, draw):
    fig = plt.figure(figsize=size)
    fig.suptitle(title, x=0.02, y=0.985, ha="left", va="top", fontsize=9.5, fontweight="bold")
    draw(fig)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)
    print(f"{name}.svg  {title}")


def axes(fig, rect=(0.16, 0.14, 0.8, 0.74), grid="y"):
    ax = fig.add_axes(rect)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid)
    return ax


def log_cg(ax, labels=(True, True)):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_xticks(10.0 ** np.arange(0, 7)); ax.xaxis.set_major_formatter(POW10); ax.yaxis.set_major_formatter(POW10)
    ax.set_xlabel(C_LABEL if labels[0] else ""); ax.set_ylabel(G_LABEL if labels[1] else "")
    ax.tick_params(labelbottom=labels[0], labelleft=labels[1])


def colourbar(fig, rect, norm, cmap, label, **kw):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, cmap), cax=fig.add_axes(rect), **kw)
    cb.outline.set_visible(False); cb.set_label(label)
    return cb


def bins():
    B = T["bins"]
    return B[(B.C > 0) & (B.G > 0)]


def final(method, target, run, setup="x1"):
    """(code, constants fitted on train) of a run's final equation."""
    F = T["finals"]
    r = F[(F.method == method) & (F.target == target) & (F.setup == setup) & (F.run == run)].iloc[0]
    return r.code, json.loads(r.params)


# ------------------------------------------------------------------ the search, and every final equation on the bins
def progress(fig, method, target):
    d = T["progress"]; d = d[d.target == target]
    lo, hi = min(d.share.min() - 0.02, 0), d.share.max() + 0.02     # the same y range for both methods
    d, ax = d[d.method == method], axes(fig)
    if method == "llm":
        med = d.groupby("iteration").share.median()
    else:                              # PySR prints at uneven iterations: each run's latest, from where every run has printed
        grid = np.arange(d.groupby("run").iteration.min().max() // 10 * 10 + 10, 101, 10)
        med = pd.Series([d[d.iteration <= x].sort_values("iteration").groupby("run").share.last().median() for x in grid], grid)
    for _, r in d.groupby("run"):
        ax.plot(r.iteration, r.share, color=COL[method], alpha=0.22, lw=0.6)
    ax.plot(med.index, med.values, color=COL[method], lw=2, marker="o", ms=2.5)
    ax.annotate(f"{med.iloc[-1]:.2f}", (med.index[-1], med.iloc[-1]), xytext=(-2, 6), textcoords="offset points",
                ha="right", fontweight="bold")
    ax.set_xlim(0, med.index[-1] * 1.02); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{METHOD[method]} iteration"); ax.set_ylabel(SHARE)


def collapse(fig, method, target):
    d = T["collapse"]; d = d[(d.method == method) & (d.target == target)].dropna(subset=["predicted"])
    lo, hi = d.observed.min(), d.observed.max()
    lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
    ax = axes(fig, (0.17, 0.14, 0.74, 0.74), grid="both")
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, zorder=3)
    ax.scatter(d.predicted.clip(lo, hi), d.observed, s=5, color=COL[method], alpha=0.10, lw=0, zorder=2, rasterized=True)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("predicted " + UNIT[target]); ax.set_ylabel("observed " + UNIT[target])
    ax.text(0.04, 0.96, f"r = {np.corrcoef(d.predicted, d.observed)[0, 1]:.2f}", transform=ax.transAxes, va="top",
            fontweight="bold")


# ------------------------------------------------------------------ an equation in the parts its author wrote
def num(v):
    m, _, e = f"{v:.3g}".partition("e")
    return m + ("·10" + str(int(e)).translate(SUP) if e else "")


def sections(code, p, *inputs):
    """The program as its author divided it, nothing interpreted: every variable model() assigns that varies over
    the inputs, with the comment above it as label and its line (constants filled in) as formula. Left out:
    constants, a fixed transform of one input (no fitted constant in it: guards such as max(C, 1e-9), log(C)) and
    the result itself. Returns ([(name, label, formula, values, inputs it rests on, whether its line uses an input
    directly)], result line, output)."""
    ns, p = {"np": np, "math": math}, np.asarray(p, float)
    exec(code, ns)
    shape = np.broadcast_shapes(*[np.shape(a) for a in inputs])
    with np.errstate(all="ignore"):
        out = np.broadcast_to(np.asarray(ns["model"](*inputs, p), float), shape)
    fn = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "model"][-1]
    args = [a.arg for a in fn.args.args]
    lines, env = code.splitlines(), dict(ns, **dict(zip(args, (*inputs, p))))
    deps, fitted, scalars = {a: {a} for a in args[:-1]}, {args[-1]}, set()
    leaves = set(args[:-1])                                        # the inputs, and their fixed transforms
    raw = lambda n: re.sub(r"p\[(\d+)\]", lambda m: num(p[int(m.group(1))]), " ".join(ast.get_source_segment(code, n).split())) \
        .replace("np.", "").replace("**", "^").replace(" + 0.0 * C", "")
    text = lambda n: re.sub(r"\b[A-Za-z_]\w*\b", lambda m: num(float(env[m.group(0)])) if m.group(0) in scalars else m.group(0), raw(n))
    same = lambda a, b: np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
    found, result = [], None
    for st in fn.body:
        if isinstance(st, ast.Return):
            break
        with np.errstate(all="ignore"):
            exec(compile(ast.Module([st], []), "<model>", "exec"), env)
        used = {n.id for n in ast.walk(st.value) if isinstance(n, ast.Name)} if hasattr(st, "value") else set()
        for t in getattr(st, "targets", []):
            for n in (n for n in ast.walk(t) if isinstance(n, ast.Name)):
                deps[n.id] = set().union(*[deps.get(u, set()) for u in used])
                fitted |= {n.id} if used & fitted else set()
                scalars |= {n.id} if np.ndim(env[n.id]) == 0 else set()
        if not (isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name)):
            continue
        name = st.targets[0].id
        try:
            v = np.broadcast_to(np.asarray(env[name], float), shape)
        except ValueError:
            continue
        if np.ptp(v[np.isfinite(v)]) == 0:
            continue
        if len(deps[name]) == 1 and name not in fitted:
            leaves.add(name)
            continue
        if same(np.clip(v, np.nanmin(out), np.nanmax(out)), out):  # the result (before its final clip)
            result = text(st)
            continue
        k, note = st.lineno - 2, []
        while k >= 0 and lines[k].strip().startswith("#"):
            note.insert(0, lines[k].strip().lstrip("# ")); k -= 1
        found.append((name, " ".join(note) or name, text(st.value), v, deps[name], bool(used & leaves)))
    if result is None and isinstance(fn.body[-1], ast.Return):     # no result variable: the returned expression
        result = text(fn.body[-1].value)
    return found, result or "", out


def parts(fig, method, target, run, setup="x1"):
    """A run's final program section by section, then its prediction with the bins (white: matches, black: off).
    Maps over (C, G): the sections whose line uses C or G directly ("part 1", ...); the other sections, which only
    combine parts, are written above the prediction."""
    code, p = final(method, target, run, setup)
    c, g = np.geomspace(10 ** -0.6, 10 ** 6.2, 300), np.geomspace(0.01 / 2.2, 1000 * 2.2, 220)
    CC, GG = np.meshgrid(c, g)
    found, result, out = sections(code, p, CC, GG)
    B = bins()
    obs, pred = B[target], sections(code, p, B.C.to_numpy(), B.G.to_numpy())[2]
    drawn = [f for f in found if f[5] and f[4] & {"C", "G"}]

    def named(t):                                                  # drawn sections are called by their number
        for n, (name, *_) in enumerate(drawn):
            t = re.sub(rf"\b{name}\b", f"part {n + 1}", t)
        return t.replace(" * ", " × ").replace("*", " × ")
    wrap = lambda t, n, most: (lambda L: L[:most - 1] + [L[most - 1] + " …"] if len(L) > most else L)(textwrap.wrap(t, n))
    panels = [("\n".join(wrap(f"Part {n + 1}: {f[1]}", 48, 2) + wrap(named(f[2]), 52, 2)), f[3], f"part {n + 1}")
              for n, f in enumerate(drawn)]
    rest = [line for f in found if f not in drawn for line in wrap(f"{f[0]} = {named(f[2])}", 52, 2)]
    panels.append(("\n".join((["Prediction"] + rest + wrap(named(result), 52, 4))[:7]), out, "predicted " + UNIT[target]))
    cols = min(2, len(panels)); rows = -(-len(panels) // cols)
    W, H = 0.5 + 3.3 * cols, 0.75 + 2.75 * rows
    fig.set_size_inches(W, H)
    for k, (text, z, bar) in enumerate(panels):
        last = k == len(drawn)
        i, j = divmod(k, cols)
        x, y = (0.6 + 3.3 * j) / W, (H - 0.35 - 2.75 * (i + 1) + 0.55) / H
        ax, lo, hi = axes(fig, (x, y, 2.2 / W, 1.7 / H), grid=None), np.nanmin(z), np.nanmax(z)
        norm = LogNorm(lo, hi) if lo > 0 and hi > 10 and hi / lo > 1e3 and not last else Normalize(lo, hi)
        ax.pcolormesh(c, g, z, cmap=HEAT, norm=norm, shading="auto", rasterized=True)
        log_cg(ax, (True, j == 0 or last)); ax.set_xticks(10.0 ** np.arange(0, 7, 2))
        ax.text(0, 1.04, text, transform=ax.transAxes, fontsize=7, va="bottom", linespacing=1.3)
        cb = colourbar(fig, (x + 2.27 / W, y, 0.08 / W, 1.7 / H), norm, HEAT, bar)
        cb.ax.tick_params(labelsize=7, which="both"); cb.ax.yaxis.label.set_size(7)
        if isinstance(norm, LogNorm):
            cb.ax.minorticks_off(); cb.ax.yaxis.set_major_formatter(POW10)
    span = 0.3 * np.ptp(obs)                                       # bins on the prediction: white = matches, black = off by `span`
    miss = Normalize(0, span)
    ax.scatter(B.C, B.G, s=9, c=GREY(miss((obs - pred).abs())), edgecolor=MUTED, lw=0.4, zorder=3)
    cb = colourbar(fig, (x + 1.0 / W, y - 0.75 / H, 1.2 / W, 0.07 / H), miss, GREY, "", orientation="horizontal",
                   ticks=[0, span])
    cb.outline.set_visible(True); cb.outline.set_edgecolor(MUTED); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=7); cb.ax.set_xticklabels(["0", f"{span:.2g}"])
    fig.text(x + 0.92 / W, y - 0.715 / H, "bins: |observed − predicted|", ha="right", va="center", fontsize=7, color=INK2)


def collage(fig, method, target):
    """The predicted map of every x1 final equation, one panel per split, one colour scale (the observed bins'
    range), with its held-back share above it."""
    B = bins()
    norm = Normalize(max(B[target].min(), 0) if target == "cei" else B[target].min(), B[target].max())
    c, g = np.geomspace(10 ** -0.6, 10 ** 6.2, 120), np.geomspace(B.G.min() / 2.2, B.G.max() * 2.2, 90)
    F = T["finals"]; F = F[(F.method == method) & (F.target == target) & (F.setup == "x1")].set_index("run")
    for r in range(30):
        i, j = divmod(r, 5)
        ax = axes(fig, (0.085 + 0.16 * j, 0.815 - 0.149 * i, 0.145, 0.118), grid=None)
        if r in F.index:
            z = sections(F.code[r], json.loads(F.params[r]), *np.meshgrid(c, g))[2]
            ax.pcolormesh(c, g, z, cmap=HEAT, norm=norm, shading="auto", rasterized=True)
            ax.text(1, 1.03, f"{F.share[r]:.2f}", transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom")
        log_cg(ax, (i == 5, j == 0)); ax.set_xticks(10.0 ** np.arange(0, 7, 3)); ax.set_yticks(10.0 ** np.arange(-2, 4, 2))
        ax.set_xlim(c[0], c[-1]); ax.set_ylim(g[0], g[-1])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(labelsize=6.5)
        ax.text(0, 1.03, f"split {r}", transform=ax.transAxes, fontsize=6.5, color=INK2, va="bottom")
    fig.text(0.5, 0.012, C_LABEL, ha="center"); fig.text(0.012, 0.5, G_LABEL, rotation=90, va="center")
    colourbar(fig, (0.9, 0.3, 0.018, 0.4), norm, HEAT, "predicted " + UNIT[target])


# ------------------------------------------------------------------ complexity
def complexity(fig, target, method):
    """Held-back share against the complexity penalty at x1 (0.1 mbit per constant + 0.01 per node), every final
    equation at every penalty, the penalties as shades."""
    F = T["finals"]; F = F[(F.target == target) & F.setup.str.startswith("x")]
    ax = axes(fig, (0.13, 0.14, 0.67, 0.74), grid="both")
    pos = {s: i / (len(SCALES) - 1) for i, s in enumerate(SCALES)}
    d = F[F.method == method]
    ax.scatter(0.1 * d.k + 0.01 * d.nodes, d.share.clip(0, None), s=11, c=RAMP[method](d.scale.map(pos)), alpha=0.85, lw=0)
    ax.set_xlim(0, (0.1 * F.k + 0.01 * F.nodes).max() * 1.04); ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel(PENALTY); ax.set_ylabel(SHARE)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=4, color=RAMP[method](pos[s]), label=f"×{s:g}") for s in SCALES],
              title=METHOD[method], handletextpad=0.2, labelspacing=0.3, fontsize=7, title_fontsize=7,
              loc="upper left", bbox_to_anchor=(1.0, 1.03))


# ------------------------------------------------------------------ seed equations and prompts (Fig. 5)
SETS = [("x1", "standard\nprompt"), ("seed2", "seed equation\nwithout G/C"), ("seed3", "gradient-only\nseed equation"),
        ("nobio_object", "no biology,\nparameters\ndescribed"), ("nobio_object_vague", "no biology,\nparameters\nundescribed"),
        ("nobio_dicty_vague", "Dictyostelium\nnamed, parameters\nundescribed")]


def prompt_sets():
    F = T["finals"]
    return [(F[(F.method == "llm") & (F.target == "cei") & (F.setup == s)].sort_values("run"), lab) for s, lab in SETS]


def prompt_dots(fig):
    ax, rng = axes(fig, (0.12, 0.34, 0.84, 0.56)), np.random.default_rng(0)
    sets = prompt_sets()
    for i, (d, _) in enumerate(sets):
        v, tri = d.share.to_numpy(), d.gc.astype(bool).to_numpy()
        x = i + rng.uniform(-0.14, 0.14, len(v))
        ax.scatter(x[tri], v[tri], s=16, marker="^", color=INK, alpha=0.7, lw=0, zorder=3)
        ax.scatter(x[~tri], v[~tri], s=11, marker="o", color=INK, alpha=0.7, lw=0, zorder=3)
        ax.plot([i - 0.27, i + 0.27], [np.median(v)] * 2, color=INK, lw=1.6, zorder=4)
        ax.text(i + 0.31, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=7)
    ax.set_xticks(range(len(sets)), [f"{lab}\n(n = {len(d)})" for d, lab in sets], fontsize=7)
    ax.set_xlim(-0.5, len(sets) - 0.15); ax.set_ylabel(SHARE)
    ax.scatter([], [], marker="^", s=16, color=MUTED, label="follows G/C")
    ax.scatter([], [], marker="o", s=11, color=MUTED, label="gradient only")
    ax.legend(loc="lower left", fontsize=7, handletextpad=0.2)


def prompt_maps(fig):
    B = bins()
    norm = Normalize(max(B.cei.min(), 0), B.cei.max())
    c, g = np.geomspace(10 ** -0.6, 10 ** 6.2, 120), np.geomspace(B.G.min() / 2.2, B.G.max() * 2.2, 90)
    for j, (d, lab) in enumerate(prompt_sets()):
        v = d.share.to_numpy()
        k = int(np.argsort(v)[(len(v) - 1) // 2])                 # the median equation (lower middle for an even count)
        row = d.iloc[k]
        z = sections(row.code, json.loads(row.params), *np.meshgrid(c, g))[2]
        ax = axes(fig, (0.06 + 0.146 * j, 0.22, 0.128, 0.44), grid=None)
        ax.pcolormesh(c, g, z, cmap=HEAT, norm=norm, shading="auto", rasterized=True)
        ax.scatter(B.C, B.G, s=0.8, color="#ffffff", alpha=0.45, lw=0)
        log_cg(ax, (True, j == 0)); ax.set_xticks(10.0 ** np.arange(0, 7, 3)); ax.set_yticks(10.0 ** np.arange(-2, 4, 2))
        ax.set_xlim(c[0], c[-1]); ax.set_ylim(g[0], g[-1]); ax.set_xlabel(""); ax.tick_params(labelsize=6.5)
        if j:
            ax.set_ylabel("")
        ax.text(0, 1.04, lab, transform=ax.transAxes, fontsize=6.8, va="bottom", fontweight="bold")
        ax.text(1, -0.2, f"split {row.run}: {row.share:.2f}", transform=ax.transAxes, fontsize=6.8, ha="right", va="top")
    fig.text(0.47, 0.04, C_LABEL + "   (each set's median equation; its split and held-back share)", ha="center", fontsize=7.5)
    colourbar(fig, (0.94, 0.24, 0.012, 0.5), norm, HEAT, "predicted CEI")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    a = ap.parse_args()
    for name in ("bins", "finals", "progress", "collapse"):
        T[name] = pd.read_csv(a.results / f"{name}.csv")
    out = a.results / "figures"
    out.mkdir(exist_ok=True)
    f = lambda name, title, draw, size=(3.5, 3.0): figure(out, name, title, size, draw)
    for fig, t in (("fig2", "cei"), ("fig3", "speed"), ("S3", "costheta")):
        f(f"{fig}A", f"{NAME[t]}: LLM search", lambda x, t=t: progress(x, "llm", t))
        f(f"{fig}B", f"{NAME[t]}: LLM equations", lambda x, t=t: collapse(x, "llm", t))
        f(f"{fig}C", f"{NAME[t]}: PySR search", lambda x, t=t: progress(x, "pysr", t))
        f(f"{fig}D", f"{NAME[t]}: PySR equations", lambda x, t=t: collapse(x, "pysr", t))
    for fig, t in (("fig2", "cei"), ("fig3", "speed")):
        f(f"{fig}E", f"{NAME[t]}: an LLM equation in parts", lambda x, t=t: parts(x, "llm", t, 0))
        f(f"{fig}F", f"{NAME[t]}: a PySR equation", lambda x, t=t: parts(x, "pysr", t, 0))
    for fig, t in (("fig4", "cei"), ("S8", "speed")):
        f(f"{fig}A", f"{NAME[t]}, LLM: complexity and fit", lambda x, t=t: complexity(x, t, "llm"), (4.6, 3.0))
        f(f"{fig}B", f"{NAME[t]}, PySR: complexity and fit", lambda x, t=t: complexity(x, t, "pysr"), (4.6, 3.0))
        f(f"{fig}C", f"{NAME[t]}: an LLM equation at ×10", lambda x, t=t: parts(x, "llm", t, 0, "x10"))
        f(f"{fig}D", f"{NAME[t]}: a PySR equation at ×10", lambda x, t=t: parts(x, "pysr", t, 0, "x10"))
    f("fig5A", "CEI: LLM equations by seed equation and prompt", prompt_dots, (7.2, 3.6))
    f("fig5B", "CEI: median equation under each seed equation and prompt", prompt_maps, (8.6, 2.9))
    for fig, t, m in (("S1", "cei", "llm"), ("S2", "cei", "pysr"), ("S4", "costheta", "llm"), ("S5", "costheta", "pysr"),
                      ("S6", "speed", "llm"), ("S7", "speed", "pysr")):
        f(fig, f"{NAME[t]}: all {METHOD[m]} equations", lambda x, t=t, m=m: collage(x, m, t), (7.0, 8.0))


if __name__ == "__main__":
    main()
