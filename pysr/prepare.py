""" Prepare the data for PySR to search on, in the same way as the LLM search 
"""
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

LLM = Path(__file__).resolve().parents[1] / "llm"
sys.path[:0] = [str(LLM), str(LLM / "experiments")]
from chemoeq import data                                          # noqa: E402
import paper                                                      # noqa: E402

BIN_UM = 100.0


def bin_sets(df: pd.DataFrame, st: dict) -> pd.DataFrame:
    """The segments split by `st` (a split.json) into train / val / test, each summarised into the bins."""
    table = pd.DataFrame(st["blocks"]).T                           # condition x block -> set
    code = pd.Categorical(df["cond"].astype(str), categories=list(table.index)).codes
    assert (code >= 0).all(), "conditions that the split does not have"
    label = table.to_numpy()[code, data.block_of(df["x_mid"], BIN_UM, table.shape[1], 0.0)]
    parts = []
    for s in ("train", "val", "test"):
        b = data.bin_rows(df[label == s], BIN_UM)
        parts.append(pd.DataFrame({"set": s, "cond": b["cond"].astype(str).to_numpy(), "x_um": b["x_mid"].to_numpy(),
                                   "C": b["C"].to_numpy(), "G": b["G"].to_numpy(), "cei": b["cei"].to_numpy(),
                                   "speed": b["speed"].to_numpy(), "costheta": b["costheta"].to_numpy(),
                                   "n_segments": b["weight"].astype(int).to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def main():
    root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "data"
    out.mkdir(parents=True, exist_ok=True)
    S = paper.SETTINGS
    src = replace(S["data"], root=str(root))
    seg = out / "segments.npz"
    if not seg.exists():
        data.save(data.prepare(src), seg)
    df = data.load(seg)["df"]
    conditions = sorted(str(c) for c in df["cond"].unique())
    n_blocks = math.ceil(src.kept_range()[1] / BIN_UM)
    for r, seed in enumerate(paper.SPLIT_SEEDS):
        st = {"seed": seed, "unit": "bridge", "block_um": BIN_UM, "test_fraction": S["test_fraction"],
              "val_fraction": S["val_fraction"], "start_um": 0.0,
              "blocks": data.draw_blocks(conditions, n_blocks, S["test_fraction"], S["val_fraction"], seed)}
        (out / f"split_r{r}.json").write_text(json.dumps(st, indent=2) + "\n")
        bin_sets(df, st).to_csv(out / f"bins_r{r}.csv", index=False)
    print(f"{len(df)} segments from {len(conditions)} conditions; 30 splits of bin rows in {out}")


if __name__ == "__main__":
    main()
