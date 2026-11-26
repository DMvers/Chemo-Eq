"""Write the tracks in the same layout and units as the original ChemoTrack tracks

    python retrack/convert_tracks.py <raw tracks folder> <original tracks folder> <tracks folder>

  raw tracks folder       retrack.sh output: <cond>/<day>/<pos>/<cond>_<day>_<pos>_tracks.csv (Nr,TID,PID,x,y,t;
                          pixels, frames)
  original tracks folder  the authors' tracks of the 8 positions in stacks_5s.txt (<cond>/<day>/<pos>.csv, from
                          fetch_one.py ... csv); their image stacks hold only the first 10 min, so they are not retracked
  tracks folder           output: <cond>/<day>/<pos>.csv with Nr, TID, PID, x (µm), y (µm), t (min)

x, y: pixels × 0.91 µm (the pixel size of the 2×2-binned images). t: frame × 0.5 min. Rows sorted by track and time.
The 8 positions' files are copied unchanged.
"""
import shutil, sys
from pathlib import Path
import pandas as pd

RAW, ORIG, OUT = map(Path, sys.argv[1:4])
UM_PER_PX, MIN_PER_FRAME = 0.91, 0.5
HERE = Path(__file__).resolve().parent
done, missing = 0, []
for item in (HERE / "stacks.txt").read_text().split():
    cond, day, pos = item.split("/")
    src = RAW / cond / day / pos / f"{cond}_{day}_{pos}_tracks.csv"
    if not src.exists():
        missing.append(item)
        continue
    d = pd.read_csv(src)
    d.columns = ["Nr", "TID", "PID", "x", "y", "t"]
    d = d.sort_values(["TID", "t"], kind="stable")
    dest = OUT / cond / day / f"{pos}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Nr": range(1, len(d) + 1), "TID": d.TID.to_numpy(), "PID": d.PID.to_numpy(),
                  "x (µm)": (d.x * UM_PER_PX).round(3).to_numpy(), "y (µm)": (d.y * UM_PER_PX).round(3).to_numpy(),
                  "t (min)": (d.t * MIN_PER_FRAME).round(3).to_numpy()}).to_csv(dest, index=False)
    done += 1
for item in (HERE / "stacks_5s.txt").read_text().split():
    cond, day, pos = item.split("/")
    dest = OUT / cond / day / f"{pos}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ORIG / cond / day / f"{pos}.csv", dest)
print(f"wrote {done} retracked positions and 8 original ones to {OUT}; missing {len(missing)}: {missing}")
