"""Split a ChemoTrack image stack (one multi-page TIFF per position) into one TIFF per frame (frame_000.tif, ...), so we can load and segment them individally. Pixel values unchanged.
Usage: split_frames.py <stack.tif> <out folder>"""
import sys
from pathlib import Path
import tifffile

stack = tifffile.imread(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for k, frame in enumerate(stack):
    tifffile.imwrite(out / f"frame_{k:03d}.tif", frame)
print(sys.argv[1], stack.shape, stack.dtype, "->", out, len(stack), "frames")
