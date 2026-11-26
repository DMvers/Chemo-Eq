"""Template for one experiment

    from chemoeq import Experiment, ChemoTrack, cli
    exp = Experiment(name="...", inputs={"C": "C", "G": "G"}, system_prompt="...",
                     prompt="...", seeds=["..."], iterations=300)
    if __name__ == "__main__":
        cli(exp)

`inputs` maps each argument of the candidate program to a column of the segment
table, so model(C, G, p) gets the concentration and gradient. Blinding is a matter
of renaming: {"x1": "C", "x2": "G"} and a prompt without biology.

The prompt is a template. These placeholders are filled in by chemoeq:
    {signature}        model(C, G, p)
    {outputs}          what the program returns, e.g. (mu, kappa)
    {ranges}           one line per input: training range, whether 0 occurs
    {max_params}       the parameter limit
    {n_conditions}     number of conditions in the data
    {segment_minutes}  duration of one segment
Other braces are left alone, so code in the prompt needs no escaping.
"""
from __future__ import annotations

import hashlib
import json
import keyword
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .likelihood import MeanCEI

PROJECT = Path(__file__).resolve().parents[1]           # the chemoeq folder
RUNS = "${CHEMOEQ_RUNS:-~/chemoeq-runs}"                # results, outside chemoeq (shared by copies)
COLUMNS = ("C", "G", "x_mid", "theta_prev", "speed_prev", "cei_prev", "straight_prev", "speed", "path", "straightness")
RESERVED = {"p", "np", "numpy", "math", "model", "N_PARAMS", "PARAM_INIT", "theta", "cei", "speed_obs", "cond", "file"}


def resolve(path, base=PROJECT) -> Path:
    """Expand ${VAR:-default}, ${VAR}, $VAR and ~; relative paths are relative to
    `base` (the chemoeq folder unless given)."""
    s = re.sub(r"\$\{(\w+):-([^}]*)\}", lambda m: os.environ.get(m.group(1)) or m.group(2), str(path))
    p = Path(os.path.expanduser(os.path.expandvars(s)))
    return p if p.is_absolute() else Path(base) / p


def runs_path(path="") -> Path:
    """A path in the results folder ($CHEMOEQ_RUNS, by default ~/chemoeq-runs, next to
    chemoeq rather than inside it); absolute paths are kept as they are."""
    return resolve(path, base=resolve(RUNS))


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ------------------------------------------------------------------ data sources

@dataclass
class ChemoTrack:
    """Tracker CSVs in <root>/<condition>/<date>/Pos*.csv (scripts/download_data.py).
    Segmentation follows the authors' analysis_cei_theta.py."""
    root: str = "${CHEMOTRACK_DATA:-~/ChemoTrack}"
    track_col: int = 1            # 0-based CSV columns; check them with `prepare`
    x_col: int = 3
    y_col: int = 4
    frame_col: int | None = None  # set it to enable gap checks and sorting
    csv_has_header: bool = True
    bridge_length_um: float = 1000.0
    frame_interval_min: float = 0.5
    seg_frames: int = 5           # frames per segment (5 frames = 2 min)
    min_path_um: float = 15.0     # minimum distance moved in a segment
    x_range_um: tuple | None = None   # (from, to): keep only segments whose midpoint is in this stretch
                                      # of the bridge, from <= x < to; None = the whole bridge
    segmentation: str = "chemoeq"     # "authors": exactly as analysis_cei_theta.py (n // seg_frames
                                      # segments per track, a track is a run of rows with the same id,
                                      # the last track of each file is not used); see data.py
    per_condition: int | str | None = None   # the same number of segments from every condition, drawn at
                                      # random: a number, or "equal" for the smallest condition's count
    per_condition_seed: int = 0       # the draw of those segments
    time_col: int | None = None       # the time of each row (minutes, or seconds when the file's largest
                                      # time is over 600), for `resample`
    resample: bool = False            # files whose frames are shorter than frame_interval_min keep only
                                      # the rows on its grid (time a multiple of it), so 5 s frames become 30 s
    c_at: str = "middle"              # where a segment's C is taken: its midpoint, or "start", its first
                                      # position (where the previous segment ended); x_mid stays the midpoint
    exclude_conditions: tuple | None = None   # condition folders left out entirely, after per_condition
                                      # (so every other condition keeps exactly the same segments)

    # settings left out of the table's name and the run's identity while they have these values,
    # so tables and runs made before the settings existed keep their names
    _UNUSED = {"x_range_um": None, "segmentation": "chemoeq", "per_condition": None, "time_col": None,
               "resample": False, "c_at": "middle", "exclude_conditions": None}

    def folder(self) -> Path:
        return resolve(self.root)

    def ensure(self, log=print):
        if not self.folder().is_dir():
            raise FileNotFoundError(f"no data folder {self.folder()} (root = {self.root!r}). "
                                    "Download it with: python scripts/download_data.py")

    def key(self) -> str:
        """Names the cached segment table: the settings and where the files are."""
        # schema 2: segments also carry straight_prev (cached tables made before it are not reused)
        return f"{type(self).__name__.lower()}-{_hash({**self._settings(), 'root': str(self.folder()), 'schema': 2})}"

    def identity(self) -> dict:
        """The settings without the location, so moving the data keeps runs resumable."""
        d = self._settings()
        d.pop("root")
        return {"type": type(self).__name__, **d}

    def _settings(self) -> dict:
        d = asdict(self)
        for k, unused in self._UNUSED.items():
            if d.get(k) == unused:
                d.pop(k, None)
        if d.get("per_condition") is None:
            d.pop("per_condition_seed", None)
        if "x_range_um" in d:
            d["x_range_um"] = [float(v) for v in d["x_range_um"]]
        return d

    def kept_range(self) -> tuple:
        """The stretch of the bridge the segments come from, in um."""
        lo, hi = self.x_range_um if self.x_range_um is not None else (0.0, self.bridge_length_um)
        return float(lo), float(hi)


# ------------------------------------------------------------------ language model

@dataclass
class LLM:
    """How to reach the local model server (scripts/serve_qwen.sh)."""
    model: str = "Qwen/Qwen3.8-27B"
    uds: str | None = "auto"      # Unix socket ~/.chemoeq/vllm.sock; None -> base_url over TCP
    base_url: str = "http://127.0.0.1:8000/v1"
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int | None = None              # None: not sent (the server's default, every token)
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int = 2048                # the reply, reasoning included
    disable_thinking: bool = True
    reasoning_effort: str | None = None   # with thinking: "xhigh" (Qwen3.8's default), "medium", "low"
    context_tokens: int | None = None     # the server's --max-model-len; with the fit tool, each turn's max_tokens
                                          # is lowered so the conversation so far plus the reply fits in it
    timeout_s: float = 300
    api_key: str = "EMPTY"
    retries: int = 5              # failed batches in a row before the search stops


# ------------------------------------------------------------------ prompt parts with no biology in them

EXAMPLE_TEMPLATE = """\
### Previous model {rank} (held-out score {score:.4f} bits; train gain {train_gain:.4f}; params {n_params}; nodes {complexity})
```python
{code}
```
Fitted parameters: {params}"""

DIAGNOSTICS_TEMPLATE = """\
Per-condition check of the best model above on the *training* conditions
(mean {observable} observed vs predicted{sign_note}):
{table}"""

BINNED_DIAGNOSTICS_TEMPLATE = """\
Check of the best model above on the *training* data, each gradient condition in
concentration bins (mean {observable} observed vs predicted{sign_note}):
{table}"""

ASK = """\
Propose ONE new model that you expect to score higher than all of the above.
Prefer changing the functional form or the mechanism over re-tuning constants,
and keep it as simple as the data allow. One- or two-line idea, then the code."""


# ------------------------------------------------------------------ the experiment

@dataclass
class Experiment:
    name: str
    inputs: dict | list                       # program argument -> data column
    system_prompt: str
    prompt: str
    seeds: list
    data: ChemoTrack = field(default_factory=ChemoTrack)
    likelihood: object = field(default_factory=MeanCEI)
    # Whole conditions are held out of the search: never used until `evaluate` (test),
    # or used only to score candidates (val). Everything else is used to fit parameters.
    # By default the conditions are drawn at random when a run first needs them and
    # kept in <run folder>/split.json, so resuming, `evaluate` and the figures all use
    # the same split. Give `test` and `val` as lists of condition names to fix them.
    test: list | None = None
    val: list | None = None
    test_fraction: float = 0.25               # of the conditions, when drawn at random
    val_fraction: float = 0.25
    split_unit: str = "condition"             # what is assigned to a set: whole conditions, or "bridge":
    split_block_um: float = 100.0             # blocks of this length along the bridge within each
                                              # condition, so that every condition is in all three sets
    split_block_origin_um: float | None = None   # blocks are counted from this point of the bridge (0 = the
                                              # Cmin end; blocks cut by the ends of the data are shorter);
                                              # None: from the start of the data's stretch (x_range_um)
    split_seed: int | None = None             # None: a new random split for every run
    bin_rows_um: float | None = None          # each set summarised into one row per condition and bin of this
                                              # width along the bridge (a segment goes to the nearest of the
                                              # bin centres w/2, 3w/2, ...), at the centre's concentration, with
                                              # the bin's mean CEI and its number of segments as the row's weight;
                                              # 100 = the ChemoTrack preprint's Fig. 3D bins. Only for CEI
                                              # likelihoods (log density linear in the CEI); None: every segment
    bin_weights: str = "segments"             # with bin rows, what a row counts for in fits and scores: "segments",
                                              # its number of segments (the same as fitting the segments at the
                                              # row's C); "equal", every row the same (weight 1: scores are the
                                              # mean over rows of the log density of the row's mean)
    only_with_predecessor: bool = True        # same rows whatever the inputs, so scores compare
    # search
    iterations: int = 300
    batch: int = 8                            # proposals per iteration, sent concurrently
    islands: int = 4
    island_size: int = 20
    examples_per_prompt: int = 3
    rank_temperature: float = 3.0             # examples drawn with weight exp(-rank / T)
    reset_every: int = 50                     # iterations between resets of the weaker islands
    overlap: bool = True                      # LLM writes k+1 while k is fitted (prompts lag one
                                              # iteration); False: each prompt sees all results so far
    save_prompts: bool = True                 # every prompt and reply in prompts.jsonl (~10 kB each)
    fit_rows_per_condition: int | None = 12000   # balanced subsample used during search: at most
    val_rows_per_condition: int | None = 12000   # this many segments per condition; None: every segment
    show_diagnostics: bool = True             # per-condition table in the prompt
    diagnostics_by: str = "condition"         # lines of that table: one per condition, or "concentration":
    diagnostics_bins: int = 4                 # each gradient condition in up to this many concentration
    diagnostics_span: float = 30.0            # bins (log-spaced; the lowest ends at Cmax / span), each
    diagnostics_rows_per_bin: int | None = 2000   # line from at most this many fitting segments;
    diagnostics_min_rows: int = 200           # bins with fewer segments than this join a neighbour
    random_seed: int = 0
    example_template: str = EXAMPLE_TEMPLATE
    diagnostics_template: str = DIAGNOSTICS_TEMPLATE
    ask: str = ASK
    # scoring and fitting
    max_params: int = 8
    complexity_weight: float = 1e-5           # bits/segment per syntax-tree node
    param_weight: float = 1e-4                # bits/segment per parameter
    op_costs: dict | None = None              # extra nodes per use of a function, e.g. candidate.expansion_costs():
                                              # np.tanh counts as 1 - 2/(1 + np.exp(2x)); None: every node is 1
    fit_restarts: int = 4
    fit_parallel_restarts: bool = True        # during the search, every start of a fit in a worker of its
                                              # own (8 candidates x 4 starts = 32 jobs), then one job per
                                              # candidate to finish it; False: one worker per candidate
    fit_maxiter: int = 150
    fit_timeout_s: float = 900                # per candidate (per job when the starts run in parallel); a fit
                                              # on every segment takes ~1-2 min, on the default subsample a few seconds
    fit_workers: int | str = "auto"           # CPU cores minus 2, at most 32
    fit_memory_gb: float | str = "auto"       # per worker; 0 = no cap
    # held-out evaluation
    final_full_fit: bool = True               # when a search ends: refit its best program on all
                                              # conditions and segments, shown next to the normal fit
    eval_fit_restarts: int = 8
    eval_fit_maxiter: int = 300
    eval_fit_timeout_s: float = 3600
    # figures
    llm: LLM = field(default_factory=LLM)
    out_dir: str | None = None                # default <results folder>/<name>; relative paths
                                              # are inside the results folder ($CHEMOEQ_RUNS)
    cache_dir: str = "data/prepared"          # where prepared segment tables are cached
    source: str | None = field(default=None, repr=False)   # the experiment file (set automatically)

    def __post_init__(self):
        if isinstance(self.inputs, (list, tuple)):
            self.inputs = {c: c for c in self.inputs}
        self.inputs = dict(self.inputs)
        if self.source is None:
            try:
                self.source = str(Path(sys._getframe(2).f_code.co_filename).resolve())
            except (ValueError, AttributeError):
                self.source = None
        problems = []
        xr = getattr(self.data, "x_range_um", None)
        if xr is not None and not (len(xr) == 2 and 0 <= xr[0] < xr[1] <= self.data.bridge_length_um):
            problems.append(f"data x_range_um {xr!r}: give (from, to) in um with 0 <= from < to <= the bridge length")
        if getattr(self.data, "segmentation", "chemoeq") not in ("chemoeq", "authors"):
            problems.append(f"data segmentation {self.data.segmentation!r}: use 'chemoeq' or 'authors'")
        pc = getattr(self.data, "per_condition", None)
        if not (pc is None or pc == "equal" or (isinstance(pc, int) and not isinstance(pc, bool) and pc > 0)):
            problems.append(f"data per_condition {pc!r}: a positive number, 'equal' or None")
        if getattr(self.data, "c_at", "middle") not in ("middle", "start"):
            problems.append(f"data c_at {self.data.c_at!r}: use 'middle' or 'start'")
        if getattr(self.data, "resample", False) and getattr(self.data, "time_col", None) is None:
            problems.append("data resample needs time_col, the column with each row's time")
        if not re.fullmatch(r"[A-Za-z][\w-]*", self.name):
            problems.append(f"name {self.name!r}: use letters, digits, _ and -")
        for arg, col in self.inputs.items():
            if not arg.isidentifier() or keyword.iskeyword(arg) or arg in RESERVED or arg.startswith("_"):
                problems.append(f"input name {arg!r} can't be a program argument")
            if col not in COLUMNS:
                problems.append(f"input {arg!r}: unknown column {col!r} (choose from {COLUMNS})")
        if not self.seeds:
            problems.append("give at least one seed program")
        if self.diagnostics_by not in ("condition", "concentration"):
            problems.append(f"diagnostics_by {self.diagnostics_by!r}: use 'condition' or 'concentration'")
        if self.split_unit not in ("condition", "bridge"):
            problems.append(f"split_unit {self.split_unit!r}: use 'condition' or 'bridge'")
        if self.split_unit == "bridge" and not self.random_split:
            problems.append("split_unit='bridge' draws the split at random: leave test and val unset")
        if self.split_unit == "bridge" and not self.split_block_um > 0:
            problems.append("split_block_um must be positive")
        if self.split_block_origin_um is not None and not self.split_block_origin_um <= self.data.kept_range()[0]:
            problems.append("split_block_origin_um must be at or before the start of the data's stretch of the bridge")
        if self.bin_weights not in ("segments", "equal"):
            problems.append("bin_weights must be 'segments' or 'equal'")
        elif self.bin_weights == "equal" and self.bin_rows_um is None:
            problems.append("bin_weights='equal' needs bin rows (bin_rows_um)")
        if self.bin_rows_um is not None:
            from .likelihood import CEI, Speed
            w = self.bin_rows_um
            if not w > 0:
                problems.append("bin_rows_um must be positive")
            elif not isinstance(getattr(self.likelihood, "base", self.likelihood), (CEI, Speed)):
                problems.append("bin_rows_um needs a CEI or Speed likelihood (CEI, MeanCEI, Speed): its log density "
                                "is linear in what it scores, so a bin's mean and its number of segments give its "
                                "exact likelihood")
            elif getattr(self.data, "c_at", "middle") != "middle":
                problems.append("bin_rows_um needs data c_at='middle' (C at the segment's midpoint)")
            elif self.split_unit == "bridge" and not (self.split_block_origin_um == 0
                                                      and abs(self.split_block_um / w - round(self.split_block_um / w)) < 1e-9):
                problems.append("bin_rows_um: the split's blocks must start at 0 and be a whole number of bins long, "
                                "so that every bin lies in one block")
        if self.random_split:
            if not (0 <= self.test_fraction < 1 and 0 <= self.val_fraction < 1
                    and self.test_fraction + self.val_fraction < 1):
                problems.append("test_fraction and val_fraction: each from 0 up to 1, together below 1")
        else:
            self.test, self.val = list(self.test or []), list(self.val or [])
            if set(self.test) & set(self.val):
                problems.append(f"conditions in both test and val: {sorted(set(self.test) & set(self.val))}")
        if problems:
            raise ValueError("experiment " + self.name + ":\n  " + "\n  ".join(problems))

    # -------------------------------------------------------------- derived
    @property
    def args(self) -> tuple:
        return tuple(self.inputs)

    @property
    def signature(self) -> str:
        return f"model({', '.join(self.args + ('p',))})"

    @property
    def needs_predecessor(self) -> bool:
        return self.only_with_predecessor or any(c.endswith("_prev") for c in self.inputs.values())

    @property
    def random_split(self) -> bool:
        return self.test is None and self.val is None

    @property
    def run_dir(self) -> Path:
        return runs_path(self.out_dir or self.name)

    @property
    def prepared_path(self) -> Path:
        return resolve(self.cache_dir) / f"{self.data.key()}.npz"

    def fingerprint(self) -> str:
        """Settings that make an old checkpoint invalid if they change (the prompt,
        iterations and figures may change between resumes)."""
        split = ([self.test_fraction, self.val_fraction, self.split_seed] if self.random_split
                 else [self.test, self.val])
        if self.split_unit != "condition":
            split += [self.split_unit, self.split_block_um]
            if self.split_block_origin_um is not None:
                split.append(float(self.split_block_origin_um))
        keys = {"data": self.data.identity(), "split": split, "inputs": self.inputs,
                "pred": self.needs_predecessor, "lik": repr(self.likelihood),
                "rows": [self.fit_rows_per_condition, self.val_rows_per_condition],
                "score": [self.max_params, self.complexity_weight, self.param_weight],
                "islands": self.islands, "seed": self.random_seed}
        if self.bin_rows_um is not None:          # left out otherwise, so older runs keep their identity
            keys["bins"] = float(self.bin_rows_um)
        if self.bin_weights != "segments":        # likewise
            keys["bin_weights"] = self.bin_weights
        if self.op_costs:                         # likewise
            keys["ops"] = sorted(self.op_costs.items())
        return _hash(keys)

    def scoring(self) -> dict:
        """What fitting and scoring need, small enough to send to worker processes."""
        return {"args": self.args, "likelihood": self.likelihood, "max_params": self.max_params,
                "fit_restarts": self.fit_restarts, "fit_maxiter": self.fit_maxiter,
                "fit_timeout_s": self.fit_timeout_s, "complexity_weight": self.complexity_weight,
                "param_weight": self.param_weight, "parallel_restarts": self.fit_parallel_restarts,
                "op_costs": self.op_costs}

    # -------------------------------------------------------------- prompt
    def render(self, examples: list, summary: dict, conditions: dict | None = None) -> list[dict]:
        """Chat messages for one proposal."""
        fill = {"signature": self.signature, "max_params": str(self.max_params),
                "outputs": (self.likelihood.outputs[0] if len(self.likelihood.outputs) == 1
                            else "(" + ", ".join(self.likelihood.outputs) + ")"),
                "ranges": summary["ranges"], "n_conditions": str(summary["n_conditions"]),
                "segment_minutes": f"{(self.data.seg_frames - 1) * self.data.frame_interval_min:g}"}
        text = re.sub(r"\{(" + "|".join(fill) + r")\}", lambda m: fill[m.group(1)], self.prompt.strip())
        parts = [text]
        ranked = sorted(examples, key=lambda e: e["score"])            # best last
        for i, e in enumerate(ranked, 1):
            parts.append(self.example_template.format(
                rank=i, score=e["score"], train_gain=e["train_gain"], n_params=e["n_params"],
                complexity=e["complexity"], code=e["code"].strip(),
                params="[" + ", ".join(f"{v:.4g}" for v in e["params"]) + "]"))
        if self.show_diagnostics and ranked and ranked[-1].get("diagnostics"):
            template = self.diagnostics_template
            if "order" in summary["by_condition"] and template == DIAGNOSTICS_TEMPLATE:
                template = BINNED_DIAGNOSTICS_TEMPLATE
            parts.append(template.format(
                observable=getattr(self.likelihood, "observable", "cos(theta)"),
                sign_note=getattr(self.likelihood, "sign_note", "; positive = net movement towards +x"),
                table=diagnostics_table(ranked[-1]["diagnostics"], summary["by_condition"])))
        parts.append(self.ask.strip())
        return [{"role": "system", "content": self.system_prompt.strip()},
                {"role": "user", "content": "\n\n".join(parts)}]


def input_summary(exp: Experiment, X: dict, n_conditions: int, feedback: tuple | None = None) -> dict:
    """Ranges of each program input on the training rows, overall and per condition,
    labelled only with the program's own argument names. `feedback` (arrays, order,
    owner; see search.feedback_data) makes the per-line ranges those of the feedback
    table's bins instead of whole conditions."""
    import numpy as np
    lines, by_cond = [], {}
    for a in exp.args:
        v = X[a][np.isfinite(X[a])]
        if v.size == 0:
            continue
        zero = "exactly 0 occurs" if v.min() == 0 else "can approach 0" if v.min() >= 0 else "can be negative"
        lines.append(f"  {a}: training range {v.min():.3g} to {v.max():.3g}; {zero}")
    shown = [a for a, c in exp.inputs.items() if not c.endswith("_prev") and not c.startswith("#")]
    extra = {}
    if feedback is not None:
        X, extra["order"], extra["owner"] = feedback
    for c in np.unique(X["cond"]):
        m = X["cond"] == c
        cells = []
        for a in shown:
            lo, hi = float(np.nanmin(X[a][m])), float(np.nanmax(X[a][m]))
            cells.append(f"{lo:.4g}" if abs(hi - lo) <= 1e-9 * max(1, abs(hi)) else f"{lo:.4g}..{hi:.4g}")
        by_cond[str(c)] = cells
    return {"ranges": "\n".join(lines), "n_conditions": n_conditions,
            "by_condition": {"header": shown, "rows": by_cond, **extra,
                             "format": getattr(exp.likelihood, "number_format", "+.3f")}}


def diagnostics_table(rows: list, by_condition: dict) -> str:
    head = by_condition["header"]
    width = [max(len(h), *(len(by_condition["rows"].get(r["cond"], [""] * len(head))[i]) for r in rows), 4)
             for i, h in enumerate(head)]
    def order(r):
        cells = by_condition["rows"].get(r["cond"])
        try:
            return float(cells[0].split("..")[-1]) if cells else 0.0
        except ValueError:
            return 0.0
    lines = ["  " + "  ".join(h.ljust(w) for h, w in zip(head, width)) + "  n       obs     pred"]
    fmt = by_condition.get("format", "+.3f")
    if "order" in by_condition:                 # bins of a condition together, a blank line between conditions
        by_name, owner = {r["cond"]: r for r in rows}, by_condition.get("owner", {})
        rows, previous = [by_name[k] for k in by_condition["order"] if k in by_name], None
    else:
        rows, owner, previous = sorted(rows, key=order), {}, None
    for r in rows:
        if owner and previous is not None and owner.get(r["cond"]) != previous:
            lines.append("")
        previous = owner.get(r["cond"])
        cells = by_condition["rows"].get(r["cond"], ["?"] * len(head))
        lines.append("  " + "  ".join(c.ljust(w) for c, w in zip(cells, width))
                     + f"  {r['n']:<6d} {format(r['obs_cos'], fmt):>6s}  {format(r['pred_cos'], fmt):>6s}")
    return "\n".join(lines)
