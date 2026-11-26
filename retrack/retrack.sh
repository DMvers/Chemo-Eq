#!/usr/bin/env bash
# Retrack the ChemoTrack image stacks: download each TIFF from the BioImage Archive (S-BIAD3674), split it into
# frames, then segment with Cellpose and track with Ultrack through track.py.
#
#   bash retrack/retrack.sh <images folder> <raw tracks folder> [stack list] [cellpose model]
#
#   images folder      where the TIFFs are downloaded (<cond>/<day>/<pos>.tif; about 1 TB for all)
#   raw tracks folder  output: <cond>/<day>/<pos>/<cond>_<day>_<pos>_tracks.csv (pixels, frames)
#   stack list         default retrack/stacks.txt (481 stacks); split it to run several workers at once
#   cellpose model     default retrack/model/Dictynov2026
#
# Python: $PYTHON (default retrack/.venv/bin/python). The 8 positions recorded every 5 s (stacks_5s.txt) are not
# retracked: convert_tracks.py uses the authors' tracks for them.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=$(realpath -m "${1:?images folder}"); OUT=$(realpath -m "${2:?raw tracks folder}")
LIST=$(realpath -m "${3:-$HERE/stacks.txt}"); MODEL=$(realpath -m "${4:-$HERE/model/Dictynov2026}")
PY=${PYTHON:-$HERE/.venv/bin/python}
[ -f "$MODEL" ] || { echo "no Cellpose model at $MODEL"; exit 1; }
while read -r item; do
  [ -z "$item" ] && continue
  name=${item//\//_}; work=$OUT/$item
  [ -s "$work/${name}_tracks.csv" ] && { echo "== $item already tracked"; continue; }
  echo "== $item $(date '+%F %T')"
  "$PY" "$HERE/fetch_one.py" "$IMG" "$item" || { echo "!! $item download failed"; continue; }
  mkdir -p "$work"; rm -rf "$work/frames"
  "$PY" "$HERE/split_frames.py" "$IMG/$item.tif" "$work/frames" || { echo "!! $item split failed"; continue; }
  # Ultrack keeps its database in the working folder, so each stack runs in its own
  if ( cd "$work" && "$PY" "$HERE/track.py" "$work/frames" "$MODEL" "$work/${name}_tracks.csv" > "$work/track.log" 2>&1 ); then
    rm -rf "$work/frames"; tail -1 "$work/track.log"
  else
    echo "!! $item tracking failed, see $work/track.log"
  fi
done < "$LIST"
