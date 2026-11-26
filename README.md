# Chemo-Eq

Code for the paper "Understanding chemotaxis through AI-guided equation discovery from quantitative experimental data". This repository is divided into four parts:

| Step | Folder | Input | Output |
|---|---|---|---|
| 1. Retracking | `retrack/` | ChemoTrack image stacks | tracks in µm and minutes |
| 2. PySR search | `pysr/` | the tracks from 1 | one final equation per search |
| 3. LLM search | `llm/` | the tracks from 1 | as 2, but with equations proposed by Qwen3.8-27B instead of PySR |
| 4. Scores and figures | `analysis/` | the results of steps 2 and 3 | `results/*.csv`, `results/figures/` |

Both searches use the same bins, likelihood, complexity penalty and 30 train/validation/held-back splits: split *r*
holds back the same bins in both (seeds in `llm/experiments/paper.py`).

## 1. Retracking

Requirements:
- An NVIDIA GPU with at least 4 GB free.
- About 1 TB of disk space for the images
- Python 3.10–3.12.

Install:

```bash
python3 -m venv retrack/.venv
retrack/.venv/bin/pip install -r retrack/requirements.txt     # cellpose 4.1.1, ultrack 0.6.3, tifffile; pulls in torch
retrack/.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

- If the third step prints False, install the torch build for your CUDA version first (pytorch.org), then rerun the pip line.
- The fine-tuned Cellpose model, Dictynov2026 (1.2 GB), is attached to this repository's GitHub release. Download it to `retrack/model/Dictynov2026`.
- Ultrack picks its own solver for the linking step: Gurobi if licensed, otherwise the free CBC. CBC is much slower.

Run:

```bash
bash retrack/retrack.sh <images> <raw tracks>                                     # 481 stacks
while read i; do retrack/.venv/bin/python retrack/fetch_one.py <original tracks> $i csv; done < retrack/stacks_5s.txt
retrack/.venv/bin/python retrack/convert_tracks.py <raw tracks> <original tracks> <tracks>
```

- The 8 positions recorded every 5 s (`stacks_5s.txt`) use the authors' tracks: their image stacks hold only the first 10 min.

## 2. PySR

```bash
python3 -m venv pysr/.venv && pysr/.venv/bin/pip install -r pysr/requirements.txt   # Julia installs on first import
bash pysr/run.sh <tracks>
```

- CEI and speed at penalty ×0.01–×100, cos θ at ×1; 30 splits each.
- Results: `pysr/runs/<target>_x<penalty>_r<split>/`. PySR is not bit-for-bit reproducible.

## 3. LLM search
Assumes one ~80 GB Nvidia GPU (we used an A100) and AlmaLinux.

Setting up Qwen:
- `install.sh` installs vLLM 0.31.0 in `~/venvs/vllm` and downloads the Qwen3.8-27B weights to `~/models`.
- `prelaunch.sh` starts vLLM with the paper's settings (`llm/scripts/serve_qwen.sh`)
- After waiting a few minutes for the LLM to be loaded, you can start the runs with `run_all.sh`

```bash
sudo bash llm/scripts/install.sh && sudo reboot     # once: the NVIDIA driver
bash llm/scripts/install.sh                         # vLLM, Python, model weights
bash llm/scripts/prelaunch.sh                       # start the model server
bash llm/run_all.sh <tracks>
```

- By default, all LLM runs from the paper are performed sequentially: one run each on 30 different splits of the data for CEI and speed at complexity penalty ×0.01–×100, and also cos θ at ×1, and finally 2 alternative seed equations and 3 alternative prompts (`llm/experiments/paper.py`, `llm/experiments/prompts/`).
- Results are saved in `~/chemoeq-runs/paper/<search>_r<split>/`

## 4. Scores and figures

```bash
llm/.venv/bin/pip install -r analysis/requirements.txt
llm/.venv/bin/python analysis/score.py <tracks>
llm/.venv/bin/python analysis/figures.py
```