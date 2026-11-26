"""Segment and track one image stack: Cellpose (fine-tuned model) for segmenting, Ultrack for tracking

    python retrack/track.py <frames folder> <cellpose model> <out.csv>

<frames folder> holds one TIFF per frame (split_frames.py). Output: one row per cell per frame,
Nr, TID, PID, x, y, t (x and y in pixels, rounded; t = frame number, from 1).

Needs a GPU for Cellpose to run efficiently
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from dask.array.image import imread
from ultrack import track, to_tracks_layer
from ultrack.config import MainConfig
from ultrack.imgproc import normalize
from ultrack.imgproc.segmentation import Cellpose
from ultrack.utils import labels_to_edges
from ultrack.utils.array import array_apply

CELLPROB, FLOW, MIN_AREA, LINK_PX = 0.0, 1.4, 32, 48.4


def main():
    frames, model, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    image = imread(str(frames / "*.tif"))
    cellpose = Cellpose(pretrained_model=model, gpu=True)
    labels = np.zeros(image.shape, dtype=np.int32)
    array_apply(image, out_array=labels, gamma=0.5,
                func=lambda f, gamma: cellpose(normalize(np.asarray(f), gamma=gamma), normalize=True,
                                               flow_threshold=FLOW, cellprob_threshold=CELLPROB))
    detection, contours = labels_to_edges(labels, sigma=5.0)

    cfg = MainConfig()
    n = min(20, os.cpu_count() or 1)
    cfg.data_config.n_workers = cfg.segmentation_config.n_workers = cfg.linking_config.n_workers = n
    cfg.segmentation_config.min_area = MIN_AREA
    cfg.segmentation_config.max_area = 1_000_000
    cfg.segmentation_config.threshold = 0.001
    cfg.segmentation_config.min_frontier = 0.1
    cfg.linking_config.max_neighbors = 5
    cfg.linking_config.max_distance = LINK_PX
    cfg.tracking_config.appear_weight = -0.02
    cfg.tracking_config.disappear_weight = -0.1
    cfg.tracking_config.division_weight = -0.03
    track(cfg, detection=detection, edges=contours, overwrite=True)

    t = to_tracks_layer(cfg)[0].sort_values(["track_id", "t"]).reset_index(drop=True)
    pd.DataFrame({"Nr": np.arange(1, len(t) + 1), "TID": t.track_id.astype(int),
                  "PID": t.groupby("track_id").cumcount() + 1,
                  "x": t.x.round().astype(int), "y": t.y.round().astype(int),
                  "t": (t.t + 1).astype(int)}).to_csv(out, index=False)
    print(f"{out}: {t.track_id.nunique()} tracks, {len(t)} points")


if __name__ == "__main__":      # Ultrack starts worker processes that import this file
    main()
