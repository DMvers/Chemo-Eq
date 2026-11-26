"""Fetch one ChemoTrack position <cond>/<day>/<pos> from the BioImage Archive (S-BIAD3674)
Usage: python fetch_one.py <out root> <cond/day/pos> [csv]"""
import ftplib, sys, time
from pathlib import Path

ROOT = "/biostudies/fire/S-BIAD/674/S-BIAD3674/Files/ChemoTrack"
out = Path(sys.argv[1])
cond, day, pos = sys.argv[2].split("/")
ext = "csv" if sys.argv[3:] == ["csv"] else "tif"
remote = f"{ROOT}/{cond}/{'tracks_csv' if ext == 'csv' else 'images'}/{day}/{pos}.{ext}"
target = out / cond / day / f"{pos}.{ext}"
target.parent.mkdir(parents=True, exist_ok=True)
for attempt in range(6):
    try:
        ftp = ftplib.FTP("ftp.ebi.ac.uk", timeout=120)
        ftp.login(); ftp.voidcmd("TYPE I")
        size = ftp.size(remote)
        if target.exists() and target.stat().st_size == size:
            print("have", target); break
        part = target.with_suffix(f".{ext}.part")
        with open(part, "wb") as fh:
            ftp.retrbinary(f"RETR {remote}", fh.write, blocksize=1 << 20)
        ftp.quit()
        if part.stat().st_size != size:
            raise IOError(f"size {part.stat().st_size} != {size}")
        part.rename(target)
        print("got", target, f"{size / 1e6:.0f} MB"); break
    except Exception as e:
        print("retry", attempt, type(e).__name__, e, flush=True)
        time.sleep(20)
else:
    sys.exit(f"FAILED {sys.argv[2]}")
