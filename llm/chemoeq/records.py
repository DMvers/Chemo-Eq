"""candidates.jsonl -> candidates.csv: one row per candidate, for reading by eye.

The equation column is the program written as a formula (see equation.py).
The JSONL file stays the complete record (per-condition diagnostics, full-precision
parameters, the raw reply of failed proposals). The CSV has the same candidates,
one per line, with the program code last and its line breaks written as `\\n`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .equation import one_line

COLUMNS = ["iteration", "island", "prompt", "ok", "kept", "score", "val_gain_bits", "train_gain_bits",
           "n_params", "complexity", "params", "equation", "idea", "error", "hash",
           "fingerprint", "code"]


def _num(v, fmt):
    return "" if v is None else format(v, fmt)


def row(rec: dict, outputs=("mu", "kappa")) -> dict:
    code = rec.get("code") or ""
    params = rec.get("params")
    return {
        "iteration": rec.get("iteration", ""),
        "island": rec.get("island", ""),
        "prompt": rec.get("prompt", ""),             # its line in prompts.jsonl, within the iteration
        "ok": bool(rec.get("ok")),
        "kept": "" if "kept" not in rec else bool(rec["kept"]),
        "score": _num(rec.get("score"), ".6f"),
        "val_gain_bits": _num(rec.get("val_gain"), ".6f"),
        "train_gain_bits": _num(rec.get("train_gain"), ".6f"),
        "n_params": rec.get("n_params", ""),
        "complexity": rec.get("complexity", ""),
        "params": "" if params is None else "; ".join(f"{v:.6g}" for v in params),
        "equation": one_line(code, outputs) if rec.get("ok") else "",
        "idea": " ".join((rec.get("idea") or "").split()),
        "error": " ".join((rec.get("error") or "").split()),
        "hash": rec.get("hash", ""),
        "fingerprint": rec.get("fingerprint") or "",
        "code": code.replace("\r\n", "\n").replace("\n", "\\n"),
    }


def header_ok(csv_path: str | Path) -> bool:
    """False if the CSV was written with other columns (an earlier version)."""
    with open(csv_path, encoding="utf-8-sig") as f:
        return next(csv.reader(f), []) == COLUMNS


def append(csv_path: str | Path, rec: dict, outputs=("mu", "kappa")):
    csv_path = Path(csv_path)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row(rec, outputs))


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(jsonl_path: str | Path, csv_path: str | Path | None = None,
              outputs=("mu", "kappa")) -> Path:
    """Rebuild the whole CSV from the JSONL (also works for runs made before the CSV existed)."""
    jsonl_path = Path(jsonl_path)
    csv_path = Path(csv_path) if csv_path else jsonl_path.with_suffix(".csv")
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for rec in read_jsonl(jsonl_path):
            w.writerow(row(rec, outputs))
    tmp.replace(csv_path)
    return csv_path
