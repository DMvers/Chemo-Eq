"""Turn ChemoTrack tracker CSVs into one table of migration segments.

Each row is one segment of `seg_frames` consecutive frames of one track, with
the chemical signal at the segment midpoint and the direction moved. This mirrors
the authors' analysis_cei_theta.py (5-frame segments sharing endpoints, 15 um
minimum path, linear gradient over a 1000 um bridge), so derived quantities match
the paper's CEI, but keeps every segment's angle and its predecessor.

The table is cached per data source in data/prepared/<key>.npz; `prepare` rebuilds it.
"""
from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

UNIT_NM = {"nM": 1.0, "uM": 1000.0, "mM": 1.0e6}
COND_RE = re.compile(r"Cmin-([\d.]+)(nM|uM|mM)?-Cmax-([\d.]+)(nM|uM|mM)?")

COLUMNS = [
    "cond", "file", "track", "seg", "x_mid", "C", "G", "theta", "speed",
    "path", "straightness", "theta_prev", "speed_prev", "cei_prev", "straight_prev", "has_prev",
]


def parse_condition(name: str) -> tuple[float, float]:
    """'Cmin-0-Cmax-15uM' -> (0.0, 15000.0) in nM. A missing Cmin unit takes Cmax's
    unit; no unit at all (the 'Cmin-0-Cmax-0' buffer-only control) means nM."""
    m = COND_RE.fullmatch(name)
    if not m:
        raise ValueError(f"Can't parse condition folder name: {name}")
    vmin, umin, vmax, umax = m.groups()
    umax = umax or "nM"
    return float(vmin) * UNIT_NM[umin or umax], float(vmax) * UNIT_NM[umax]


def _segments_for_track(xs, ys, frames, src):
    seg = src.seg_frames
    dt = (seg - 1) * src.frame_interval_min
    out, prev = [], None
    # Segments share endpoints, as in the authors' code; we use every complete
    # segment ((n-1)//(seg-1)) where their loop stops at n//seg (segmentation="authors": as theirs).
    n_seg = len(xs) // seg if getattr(src, "segmentation", "chemoeq") == "authors" else (len(xs) - 1) // (seg - 1)
    for j in range(n_seg):
        s, e = j * (seg - 1), j * (seg - 1) + seg - 1
        if frames is not None and frames[e] - frames[s] != seg - 1:
            prev = None                                   # gap: break the chain of predecessors
            continue
        dx, dy = xs[e] - xs[s], ys[e] - ys[s]
        path = float(np.hypot(np.diff(xs[s:e + 1]), np.diff(ys[s:e + 1])).sum())
        if path <= src.min_path_um:
            prev = None
            continue
        cur = {"seg": j, "x_mid": 0.5 * (xs[s] + xs[e]), "x_start": float(xs[s]), "theta": float(np.arctan2(dy, dx)),
               "speed": path / dt, "path": path, "straightness": float(np.hypot(dx, dy) / path), "cei": dx / path}
        if prev is not None and prev["seg"] == j - 1:
            cur["theta_prev"], cur["speed_prev"], cur["cei_prev"], cur["has_prev"] = (prev["theta"], prev["speed"],
                                                                                      prev["cei"], True)
            cur["straight_prev"] = prev["straightness"]
        else:
            cur["theta_prev"], cur["speed_prev"], cur["cei_prev"], cur["has_prev"] = np.nan, np.nan, np.nan, False
            cur["straight_prev"] = np.nan
        out.append(cur)
        prev = cur
    return out


def read_text(path: Path) -> str:
    """A CSV's text. UTF-8 (with or without a byte-order mark) if it is that, else
    Latin-1, which is how the ChemoTrack exports write their headers ("µm" as byte 0xB5)."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def read_track_csv(path: Path, src) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(read_text(path)), header=0 if src.csv_has_header else None)
    cols = {"track": src.track_col, "x": src.x_col, "y": src.y_col}
    if src.frame_col is not None:
        cols["frame"] = src.frame_col
    if getattr(src, "time_col", None) is not None:
        cols["t"] = src.time_col
    out = pd.DataFrame({k: df.iloc[:, v].to_numpy() for k, v in cols.items()})
    if getattr(src, "resample", False):
        out = resampled(out, src.frame_interval_min)
    if "frame" in out:
        # stable sort keeps the tracker's order for ties; tracks stay contiguous per id
        out = out.sort_values(["track", "frame"], kind="stable")
    return out


def resampled(t: pd.DataFrame, interval_min: float) -> pd.DataFrame:
    """A file whose frames are shorter than `interval_min`, thinned to it: only the rows whose time is a
    multiple of it are kept (5 s frames at 5, 10, ... s keep 30, 60, ... s). Times are in minutes, or in
    seconds when the file's largest time is over 600. A track keeps its rows in order, so its segments
    then span seg_frames frames of `interval_min` as in every other file. Other files are returned as
    they are."""
    tm = t["t"].to_numpy(float)
    tm = tm / 60.0 if np.nanmax(tm) > 600 else tm
    same = t["track"].to_numpy()[1:] == t["track"].to_numpy()[:-1]
    steps = np.diff(tm)[same]
    step = float(np.median(steps[steps > 0])) if (steps > 0).any() else interval_min
    k = int(round(interval_min / step))
    if k <= 1:
        return t
    return t[np.round(tm / step).astype(np.int64) % k == 0]


def preview(src, n=3) -> str:
    """First lines of one CSV next to the columns chemoeq will read from it."""
    f = min(src.folder().rglob("*.csv"), default=None)     # the same file every time
    if f is None:
        return f"no CSV files under {src.folder()}"
    head = [line.rstrip() for line in read_text(f).splitlines()[: n + 1]]
    picked = {k: getattr(src, k) for k in ("track_col", "x_col", "y_col", "frame_col")}
    return f"{f}\n  " + "\n  ".join(head) + f"\n  reading columns (0-based): {picked}"


def prepare(src, log=print) -> dict:
    """Walk <folder>/<condition>/**/*.csv and build the segment table."""
    src.ensure(log)
    root, L = src.folder(), src.bridge_length_um
    rows, conds, out_of_range, dropped_outside, unreadable = [], {}, 0, 0, []
    keep_lo, keep_hi = src.kept_range() if src.x_range_um is not None else (None, None)
    trimmed = 0
    authors = getattr(src, "segmentation", "chemoeq") == "authors"
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            cmin, cmax = parse_condition(cdir.name)
        except ValueError:
            log(f"skip {cdir.name}: not a condition folder")
            continue
        G = (cmax - cmin) / L
        conds[cdir.name] = {"cmin_nM": cmin, "cmax_nM": cmax, "G_nM_per_um": G, "n_files": 0}
        for f in sorted(cdir.rglob("*.csv")):
            fid = str(f.relative_to(root))
            try:
                t = read_track_csv(f, src)
            except Exception as e:                  # one bad file shouldn't stop the rest
                unreadable.append(fid)
                if len(unreadable) <= 10:
                    log(f"skipped {fid}: can't read it ({type(e).__name__}: {str(e)[:120]})")
                continue
            conds[cdir.name]["n_files"] += 1
            if authors:
                # as analysis_cei_theta.py: a track is a run of consecutive rows with the same id, and a
                # track is analysed when the next one starts, so the file's last track never is
                run = (t["track"] != t["track"].shift()).cumsum().to_numpy()
                keep = run < run.max()
                t, run = t[keep], run[keep]
                tracks = ((f"{g['track'].iloc[0]}.{r}", g) for r, g in t.groupby(run, sort=False))
            else:
                tracks = t.groupby("track", sort=False)       # tracker writes tracks contiguously
            for tid, g in tracks:
                xs, ys = g["x"].to_numpy(float), g["y"].to_numpy(float)
                out_of_range += int(((xs < 0) | (xs > L)).sum())
                frames = g["frame"].to_numpy(int) if "frame" in g else None
                for s in _segments_for_track(xs, ys, frames, src):
                    # off the bridge the linear-gradient formula gives C < Cmin or > Cmax
                    if not (0.0 <= s["x_mid"] <= L):
                        dropped_outside += 1
                        continue
                    # the authors' half_violin_atm.py keeps both ends open: from < x < to
                    if keep_lo is not None and not (keep_lo < s["x_mid"] < keep_hi if authors
                                                    else keep_lo <= s["x_mid"] < keep_hi):
                        trimmed += 1                 # x_range_um: outside the stretch that is kept
                        continue
                    x_c = s["x_start"] if getattr(src, "c_at", "middle") == "start" else s["x_mid"]
                    s.update(cond=cdir.name, file=fid, track=f"{fid}#{tid}", C=cmin + x_c * G, G=G)
                    rows.append(s)
    if unreadable:
        log(f"warning: {len(unreadable)} CSV file(s) could not be read and were left out"
            + (" (first 10 listed above)" if len(unreadable) > 10 else ""))
    if not rows:
        raise RuntimeError(f"No segments found under {root}"
                           + (f" ({len(unreadable)} files unreadable)" if unreadable else ""))
    if out_of_range:
        log(f"warning: {out_of_range} positions outside 0..{L} um; check bridge orientation/scale. "
            f"{dropped_outside} segments with a midpoint off the bridge were dropped.")
    if keep_lo is not None:
        log(f"x_range_um: kept the segments with a midpoint from {keep_lo:g} to {keep_hi:g} um; "
            f"{trimmed} ({trimmed / max(trimmed + len(rows), 1):.1%}) were outside it")
    df = pd.DataFrame(rows)[COLUMNS]
    if getattr(src, "per_condition", None) is not None:
        df = same_per_condition(df, src.per_condition, src.per_condition_seed, log)
    drop = set(getattr(src, "exclude_conditions", None) or ())
    if drop:
        missing = drop - set(df["cond"].unique())
        assert not missing, f"exclude_conditions not in the data: {sorted(missing)}"
        df = df[~df["cond"].isin(drop)].reset_index(drop=True)
        conds = {c: v for c, v in conds.items() if c not in drop}
        log(f"exclude_conditions: left out {', '.join(sorted(drop))}")
    for c in conds:
        conds[c]["n_segments"] = int((df["cond"] == c).sum())
    return {"df": df, "conditions": conds}


def same_per_condition(df: pd.DataFrame, n, seed: int = 0, log=print) -> pd.DataFrame:
    """`n` segments from every condition, drawn at random without replacement ("equal": as
    many as the smallest condition has), so that every condition weighs the same in every
    fit. A condition with fewer keeps all of its own. Rows keep their order in the table."""
    counts = df["cond"].value_counts()
    k = int(counts.min()) if n == "equal" else int(n)
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(df), bool)
    for c, idx in df.groupby("cond", sort=True).indices.items():
        keep[idx if len(idx) <= k else rng.choice(idx, k, replace=False)] = True
    short = counts[counts < k]
    log(f"per_condition: {k} segments from each of {len(counts)} conditions ({keep.sum()} of {len(df)} kept)"
        + (f"; fewer in {', '.join(map(str, short.index))}" if len(short) else ""))
    return df[keep].reset_index(drop=True)


def save(prepared: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {}
    for c in COLUMNS:
        v = prepared["df"][c]
        if not (pd.api.types.is_numeric_dtype(v.dtype) or pd.api.types.is_bool_dtype(v.dtype)):
            # condition and file names: a few distinct strings over millions of rows,
            # saved as integer codes plus the list of names
            cat = v if isinstance(v.dtype, pd.CategoricalDtype) else v.astype("category")
            cols["col_" + c] = cat.cat.codes.to_numpy(np.int32)
            cols["cat_" + c] = np.asarray(cat.cat.categories, dtype=str)
        else:
            cols["col_" + c] = v.to_numpy()
    np.savez_compressed(path, **cols)
    path.with_suffix(".conditions.json").write_text(json.dumps(prepared["conditions"], indent=2))


def load(path: Path) -> dict:
    z = np.load(path)
    cols = {}
    for c in COLUMNS:
        if "col_" + c not in z.files:             # a table saved before the column existed
            cols[c] = np.full(len(z["col_C"]), np.nan)
            continue
        v = z["col_" + c]
        if "cat_" + c in z.files:                # names kept as categories: small in memory
            v = pd.Categorical.from_codes(v, [str(n) for n in z["cat_" + c]])
        elif v.dtype.kind in "US":               # a table saved by an earlier version: plain strings
            names, codes = np.unique(v, return_inverse=True)
            v = pd.Categorical.from_codes(codes.ravel(), [str(n) for n in names])
        cols[c] = v
    df = pd.DataFrame(cols)
    df["has_prev"] = df["has_prev"].astype(bool)
    return {"df": df, "conditions": json.loads(path.with_suffix(".conditions.json").read_text())}


def load_for(exp, log=print) -> dict:
    """The experiment's segment table: from the cache, or prepared now."""
    if not exp.prepared_path.exists():
        save(prepare(exp.data, log), exp.prepared_path)
    return load(exp.prepared_path)


def split(df: pd.DataFrame, exp, log=print) -> dict[str, pd.DataFrame]:
    """Split by blocks of the bridge within each condition (split_unit="bridge"), then summarise each set into
    its bins (bin_rows_um)."""
    sets = _split_by_blocks(df, exp, log)
    if getattr(exp, "bin_rows_um", None) is not None:
        sets = {k: bin_rows(v, exp.bin_rows_um) for k, v in sets.items()}
    if getattr(exp, "bin_weights", "segments") == "equal":     # every bin row counts the same
        sets = {k: v.assign(weight=1.0) for k, v in sets.items()}
    return sets


def bin_rows(df: pd.DataFrame, width_um: float) -> pd.DataFrame:
    """One row per condition and bin of the bridge (a segment goes to the nearest of the bin centres
    w/2, 3w/2, ...: the ChemoTrack preprint's Fig. 3D bins for w = 100 um), placed at the centre:
    x_mid the centre and C = Cmin + G x_centre, with the bin's mean CEI (`cei`), mean speed (`speed`)
    and number of segments (`weight`). For a likelihood whose log density is linear in what it scores (CEI,
    MeanCEI, MeanCosTheta, Speed), weight times the log density of the bin's mean is the sum of the bin's segments' log
    densities at the centre's C. theta is set to 0 (the CEI is in `cei`); columns that have no meaning
    for a bin are left empty."""
    if not len(df):
        return df.assign(cei=pd.Series(dtype=float), weight=pd.Series(dtype=float))
    b = np.floor(df["x_mid"].to_numpy(float) / width_um).astype(np.int64)
    g = _bin_means(df, b)
    x = (g["bin"].to_numpy(float) + 0.5) * width_um
    return _bin_table(df, g, x, np.maximum(g["cmin"].to_numpy(float) + g["G"].to_numpy(float) * x, 0.0))


def _bin_means(df: pd.DataFrame, b: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"cond": df["cond"].to_numpy(), "bin": b, "G": df["G"].to_numpy(float),
                      "cmin": (df["C"] - df["G"] * df["x_mid"]).to_numpy(float),
                      "C": df["C"].to_numpy(float), "x": df["x_mid"].to_numpy(float),
                      "speed": df["speed"].to_numpy(float),
                      "cei": (df["straightness"] * np.cos(df["theta"])).to_numpy(float),
                      "costheta": np.cos(df["theta"]).to_numpy(float)})
    return d.groupby(["cond", "bin"], observed=True, sort=True).agg(
        G=("G", "first"), cmin=("cmin", "mean"), C=("C", "mean"), x=("x", "mean"), cei=("cei", "mean"),
        speed=("speed", "mean"), costheta=("costheta", "mean"),
        weight=("cei", "size")).reset_index()


def _bin_table(df: pd.DataFrame, g: pd.DataFrame, x: np.ndarray, C: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"cond": pd.Categorical(g["cond"], categories=df["cond"].cat.categories)
                         if hasattr(df["cond"], "cat") else g["cond"],
                         "file": "", "track": -1, "seg": -1, "x_mid": x, "C": C,
                         "G": g["G"].to_numpy(float), "theta": 0.0, "speed": g["speed"].to_numpy(float),
                         "path": np.nan, "straightness": np.nan, "theta_prev": np.nan, "speed_prev": np.nan,
                         "cei_prev": np.nan, "straight_prev": np.nan, "has_prev": False, "cei": g["cei"].to_numpy(float),
                         "costheta": g["costheta"].to_numpy(float),
                         "weight": g["weight"].to_numpy(float)})


def block_of(x_mid, block_um: float, n_blocks: int, start_um: float = 0.0) -> np.ndarray:
    """Which block of the bridge a segment's midpoint is in (0 = the Cmin end, or the
    block that begins at `start_um` when only a stretch of the bridge is used)."""
    return np.clip((np.asarray(x_mid, float) - start_um) // block_um, 0, n_blocks - 1).astype(int)


def draw_blocks(conditions: list, n_blocks: int, test_fraction: float, val_fraction: float, seed: int) -> dict:
    """Every condition's bridge blocks assigned at random: {condition: ["train", "test", ...]},
    one entry per block. Each condition gets test and val blocks in the asked proportions;
    where that is not a whole number of blocks it is rounded up or down at random, so the
    proportions hold over all conditions together."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in sorted(conditions):
        counts = []
        for f in (test_fraction, val_fraction):
            exact = f * n_blocks
            counts.append(int(exact) + int(rng.random() < exact - int(exact)))
        n_test, n_val = counts
        while n_test + n_val >= n_blocks and n_test + n_val > 0:     # leave at least one block to fit on
            if n_val >= n_test:
                n_val -= 1
            else:
                n_test -= 1
        order = rng.permutation(n_blocks)
        sets = np.full(n_blocks, "train", dtype=object)
        sets[order[:n_test]] = "test"
        sets[order[n_test:n_test + n_val]] = "val"
        out[c] = [str(s) for s in sets]
    return out


def _split_by_blocks(df: pd.DataFrame, exp, log=print) -> dict[str, pd.DataFrame]:
    """Each condition's bridge cut into blocks of `split_block_um`; the blocks, not the
    conditions, go to fit, val and test. Every condition is then in all three sets, each
    with a different stretch of its concentration range. Drawn once per run and kept in
    <run folder>/split.json, like the split by condition."""
    conditions = sorted(str(c) for c in df["cond"].unique())
    block_um = float(exp.split_block_um)
    start, end = exp.data.kept_range()                   # the whole bridge unless the data use x_range_um
    origin = getattr(exp, "split_block_origin_um", None)
    if origin is not None:                               # blocks on a grid from this point: those cut by the
        start = float(origin)                            # ends of the data are shorter
    n_blocks = max(1, int(np.ceil((end - start) / block_um)))
    path = exp.run_dir / "split.json"
    asked = {"unit": "bridge", "block_um": block_um, "test_fraction": exp.test_fraction, "val_fraction": exp.val_fraction}
    if exp.data.x_range_um is not None or origin is not None:
        asked["start_um"] = start                        # block 0 begins here
    if path.exists():
        st = json.loads(path.read_text())
        changed = [k for k, v in asked.items() if st.get(k) != v]
        if exp.split_seed is not None and st.get("seed") != exp.split_seed:
            changed.append("split_seed")
        if not changed and set(st["blocks"]) != set(conditions):
            changed.append("the conditions in the data")
        if changed:
            raise RuntimeError(f"{path} was drawn with different settings ({', '.join(changed)}). "
                               "Give the experiment a new name (or out_dir), or delete the run "
                               "folder to start over with a new split.")
    else:
        seed = exp.split_seed if exp.split_seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
        st = {"seed": seed, **asked,
              "blocks": draw_blocks(conditions, n_blocks, exp.test_fraction, exp.val_fraction, seed)}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st, indent=2) + "\n")
        os.replace(tmp, path)
        n = {k: sum(b.count(k) for b in st["blocks"].values()) for k in ("train", "val", "test")}
        log(f"random split by {block_um:g} um blocks of the bridge within each condition (seed {seed}, saved in "
            f"{path}): fit on {n['train']} blocks, val {n['val']}, test {n['test']}")
    if exp.needs_predecessor:
        df = df[df["has_prev"]]
    table = pd.DataFrame(st["blocks"]).T                 # condition x block -> set
    cond_code = pd.Categorical(df["cond"].astype(str), categories=list(table.index)).codes
    label = table.to_numpy()[cond_code, block_of(df["x_mid"], block_um, n_blocks, start)]
    return {k: df[label == k] for k in ("train", "val", "test")}


def feedback_bins(train: pd.DataFrame, conditions: dict, bins: int = 4, span: float = 30.0,
                  rows_per_bin: int | None = 2000, min_rows: int = 200, seed: int = 0,
                  x_range_um: tuple | None = None):
    """Rows for a feedback table that shows each gradient condition in concentration bins.

    A condition's fitting rows are cut into up to `bins` bins of concentration, equally
    wide on a log scale; where a condition starts at (or near) zero, the lowest bin ends
    at Cmax / span. With `x_range_um` (the stretch of the bridge the data come from) the
    bins span the concentrations of that stretch. A bin with fewer than `min_rows` rows joins its neighbour, and
    no-gradient conditions stay whole. From each bin at most `rows_per_bin` rows are
    drawn, so every line of the table rests on about the same number of segments.

    Returns the drawn rows with their bin's name in the column `cond` (so the usual
    per-condition summary gives one line per bin), the names in the order to show them,
    and the condition each name belongs to."""
    parts, order, owner = [], [], {}
    cond = train["cond"].astype(str)
    names = sorted(cond.unique(), key=lambda c: (conditions[c]["cmax_nM"], conditions[c]["G_nM_per_um"], conditions[c]["cmin_nM"]))
    for c in names:
        d = train[(cond == c).to_numpy()]
        info = conditions[c]
        lo, hi = info["cmin_nM"], info["cmax_nM"]
        if x_range_um is not None:                                 # the concentrations at the ends of the kept stretch
            lo, hi = (info["cmin_nM"] + info["G_nM_per_um"] * float(x) for x in x_range_um)
        if info["G_nM_per_um"] == 0 or bins < 2 or hi <= lo:
            groups = [d]
        else:
            start = max(lo, hi / span)
            inner = np.geomspace(start, hi, bins + (0 if lo < start else 1))[(0 if lo < start else 1):-1]
            idx = np.digitize(d["C"].to_numpy(float), inner)
            groups = []
            for g in (d[idx == k] for k in range(len(inner) + 1)):
                if groups and segments(groups[-1]) < min_rows:     # too few segments so far: join the next bin
                    groups[-1] = pd.concat([groups[-1], g])
                elif len(g):
                    groups.append(g)
            if len(groups) > 1 and segments(groups[-1]) < min_rows:
                last = groups.pop()                                # (popped first: then [-1] is the neighbour)
                groups[-1] = pd.concat([groups[-1], last])
        for k, g in enumerate(groups):
            if not len(g):
                continue
            if rows_per_bin and len(g) > rows_per_bin and "weight" not in g:   # bin rows: all of them
                g = g.sample(rows_per_bin, random_state=seed)
            name = f"{c}#{k}"
            parts.append(g.assign(cond=name))
            order.append(name); owner[name] = c
    return pd.concat(parts), order, owner


def segments(t) -> int:
    """The number of segments behind a table or arrays: rows, or the summed weights of bin rows."""
    if "weight" in t:
        return int(round(float(np.sum(t["weight"]))))
    return len(t) if isinstance(t, pd.DataFrame) else int(np.asarray(t["theta"]).size)


def subsample(df: pd.DataFrame, per_condition: int | None, seed: int) -> pd.DataFrame:
    """Balanced subsample: at most `per_condition` segments from each condition."""
    if not per_condition:
        return df
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled[shuffled.groupby("cond", observed=True).cumcount() < per_condition]


def arrays(df: pd.DataFrame, inputs: dict) -> dict[str, np.ndarray]:
    """Program inputs (by argument name) plus what scoring needs."""
    X = {arg: df[col].to_numpy(float) for arg, col in inputs.items()}
    theta = df["theta"].to_numpy(float)
    # CEI: distance moved along +x over the path length (net displacement / path = straightness)
    X.update(theta=theta, cei=df["straightness"].to_numpy(float) * np.cos(theta),
             cond=df["cond"].to_numpy(), file=df["file"].to_numpy())
    if "weight" in df:                          # bin rows (bin_rows): the bin's mean CEI and number of segments
        X.update(cei=df["cei"].to_numpy(float), weight=df["weight"].to_numpy(float))
    if "speed" in df:                           # the segment's own speed (path length / duration), for Speed
        X["speed_obs"] = df["speed"].to_numpy(float)
    # cos theta, for MeanCosTheta: a bin's mean, or a segment's own value
    X["costheta"] = df["costheta"].to_numpy(float) if "costheta" in df else np.cos(theta)
    return X
