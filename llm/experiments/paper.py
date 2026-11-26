"""Every LLM search of the paper. One variant per setup; run r of a variant uses split r (SPLIT_SEEDS[r]) and
random_seed r, so split r holds back the same bins in every variant and in PySR.

    llm/.venv/bin/python llm/experiments/paper.py cei search --repeat 0     # one search (llm/run_all.sh: all)
    llm/.venv/bin/python llm/experiments/paper.py cei prompt                # its first prompt

Variants (results in <results>/paper/<variant>_r<split>/):
  cei, speed                     penalty x1 (Fig. 2, 3)
  costheta                       penalty x1 (S3-S5 Fig)
  cei_x<s>, speed_x<s>           penalty x<s>, s = 0_01, 0_1, 10, 20, 30, 100 (Fig. 4, S8 Fig)
  cei_seed2, cei_seed3           the CEI search from another seed equation (Fig. 5)
  cei_<prompt>                   the CEI search with another prompt, from prompts/<prompt>.txt (Fig. 5)

Data: the converted tracks in $CHEMOTRACK_RETRACKED: 2-min segments (5 frames of 30 s) with a path over 15 um and
their midpoint 50-950 um along the bridge, the same number from every condition, in the 100 um bins of ChemoTrack.
Each bin mean counts once. The penalty (x1) is 1e-4 per constant and 1e-5 per syntax-tree node; a function other
than np.exp and np.log costs the nodes it takes to write it with them (OP_COSTS).
"""
import dataclasses
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the chemoeq folder

from chemoeq import LLM, ChemoTrack, MeanCEI, MeanCosTheta, Speed  # noqa: E402
from chemoeq.candidate import expansion_costs                     # noqa: E402
from chemoeq.variants import variants_cli                         # noqa: E402

SPLIT_SEEDS = [1777993453, 368656613, 1252086264, 2151410153, 3792033153, 111675376, 222626893, 1754894702,
               2737088266, 497018606, 981722328, 369894992, 1525466736, 2670497271, 1714224920, 1057972772,
               3790987429, 2590222115, 3253624066, 3319354581, 256111499, 2346912469, 3254649885, 393024230,
               1629173854, 3043321460, 22685180, 2568546724, 1336222928, 2688192519]
REPEATS = len(SPLIT_SEEDS)
REFERENCE_SPEED = 16.4          # um/min: the mean speed of the segments, the speed score's null model

SYSTEM = '''\
You propose compact, mechanistically interpretable mathematical models and write
them as short vectorised NumPy functions. Reply with a one- or two-line idea,
then exactly one ```python code block. No other text.'''

PROMPT_CEI = '''\
We are modelling how single Dicty cells steer during chemotaxis towards a
non-hydrolysable cAMP analogue. Cells migrate across a 1 mm bridge that carries a
stable, linear, one-dimensional gradient along +x. The data are from the middle
900 micrometres of the bridge.

Inputs (NumPy arrays of equal length; parameters p are fitted for you):
  C : chemoattractant concentration at the cell's position, in nM
  G : concentration gradient along +x, in nM per micrometre (G = 0 in the
      no-gradient controls)
  p : 1-D array of N_PARAMS free parameters
Training ranges:
{ranges}

What the model must return:
  For every segment of a cell's track (2 minutes) we measure the chemotactic
  efficiency index, CEI = distance moved along +x divided by the length of the
  path: -1 = straight down the gradient, +1 = straight up it, 0 = no net
  movement along x. Return {outputs}, an array (or scalar): the expected CEI of
  a segment, between -1 and 1 (values beyond +-0.995 are clipped to it).

How models are scored:
  Free parameters are fitted by maximum likelihood on training data: each
  segment's CEI is scored with the density on [-1, 1] that has the mean your
  model returns (the maximum-entropy one, proportional to exp(k * CEI)), so
  the fit matches the predicted mean CEI to the observed one.
  The score is the mean log-likelihood on *held-out data*: for every condition,
  stretches of its concentration range that were left out of the fit,
  reported as bits per observation gained over an expected CEI of 0 everywhere,
  minus a small penalty for each parameter and each syntax-tree node
  (a function other than np.exp and np.log counts as the nodes it takes to write
  it with them and arithmetic: np.tanh(x) as 1 - 2/(1 + np.exp(2*x))). A model
  that only fits the training data will score poorly; the goal is a law
  that generalises across concentrations and gradients.

Rules:
  - Define N_PARAMS (int, at most {max_params}) and PARAM_INIT, a list
    of rough starting fitting values. Use p[i] for every constant you would want fitted.
  - Use only numpy (as np) and math, through plain functions such as np.exp,
    np.log, np.maximum (no np.random or other submodules). Fully vectorised, no
    loops over observations.
  - It must return finite values for every input in the stated ranges, including 0.
  - Signature exactly: def {signature}:
  - You may transform the inputs in any way you can justify.'''

SEED_CEI = '''\
N_PARAMS = 1
PARAM_INIT = [10.0]

def model(C, G, p):
    # expected CEI proportional to the relative gradient (0 where there is no gradient)
    cei = p[0] * np.where(G > 0, G / np.maximum(C, 1e-12), 0.0)
    return cei'''

PROMPT_SPEED = '''\
We are modelling how fast single Dicty cells move during chemotaxis towards a
non-hydrolysable cAMP analogue. Cells migrate across a 1 mm bridge that carries a
stable, linear, one-dimensional gradient along +x. The data are from the middle
900 micrometres of the bridge.

Inputs (NumPy arrays of equal length; parameters p are fitted for you):
  C : chemoattractant concentration at the cell's position, in nM
  G : concentration gradient along +x, in nM per micrometre (G = 0 in the
      no-gradient controls)
  p : 1-D array of N_PARAMS free parameters
Training ranges:
{ranges}

What the model must return:
  For every segment of a cell's track (2 minutes) we measure the speed = length
  of the path divided by the duration, in micrometres per minute, whatever the
  direction. Segments with a path of 15 micrometres or less are not in the data,
  so every speed is above 7.5. Return {outputs}, an array (or scalar): the
  expected speed of a segment in micrometres per minute; it must be positive.

How models are scored:
  Free parameters are fitted by maximum likelihood on training data: each
  segment's speed is scored with a gamma distribution of shape 4 around the
  speed your model returns, so the fit matches the predicted mean speed to the
  observed one.
  The score is the mean log-likelihood on *held-out data*: for every condition,
  stretches of its concentration range that were left out of the fit,
  reported as bits per observation gained over predicting 16.4 micrometres
  per minute for every segment, minus a small penalty for each parameter and each
  syntax-tree node (a function other than np.exp and np.log counts as the nodes
  it takes to write it with them and arithmetic: np.tanh(x) as
  1 - 2/(1 + np.exp(2*x))). A model that only fits the training data will score
  poorly; the goal is a law that generalises across concentrations and gradients.

Rules:
  - Define N_PARAMS (int, at most {max_params}) and PARAM_INIT, a list
    of rough starting fitting values. Use p[i] for every constant you would want fitted.
  - Use only numpy (as np) and math, through plain functions such as np.exp,
    np.log, np.maximum (no np.random or other submodules). Fully vectorised, no
    loops over observations.
  - It must return finite values for every input in the stated ranges, including 0.
  - Signature exactly: def {signature}:
  - You may transform the inputs in any way you can justify.'''

SEED_SPEED = '''\
N_PARAMS = 1
PARAM_INIT = [13.0]

def model(C, G, p):
    # the same speed everywhere
    speed = p[0] + 0.0 * C
    return speed'''

PROMPT_COSTHETA = '''\
We are modelling how single Dicty cells steer during chemotaxis towards a
non-hydrolysable cAMP analogue. Cells migrate across a 1 mm bridge that carries a
stable, linear, one-dimensional gradient along +x. The data are from the middle
900 micrometres of the bridge.

Inputs (NumPy arrays of equal length; parameters p are fitted for you):
  C : chemoattractant concentration at the cell's position, in nM
  G : concentration gradient along +x, in nM per micrometre (G = 0 in the
      no-gradient controls)
  p : 1-D array of N_PARAMS free parameters
Training ranges:
{ranges}

What the model must return:
  For every segment of a cell's track (2 minutes) we measure the direction of
  its net displacement, theta (0 = along +x, straight up the gradient), and
  take cos(theta): +1 = straight up the gradient, -1 = straight down it,
  0 = across it, whatever the length of the path. Return {outputs}, an array
  (or scalar): the expected cos(theta) of a segment, between -1 and 1 (values
  beyond +-0.995 are clipped to it).

How models are scored:
  Free parameters are fitted by maximum likelihood on training data: each
  segment's cos(theta) is scored with the density on [-1, 1] that has the mean
  your model returns (the maximum-entropy one, proportional to
  exp(k * cos(theta))), so the fit matches the predicted mean cos(theta) to the
  observed one.
  The score is the mean log-likelihood on *held-out data*: for every condition,
  stretches of its concentration range that were left out of the fit,
  reported as bits per observation gained over an expected cos(theta) of 0
  everywhere, minus a small penalty for each parameter and each syntax-tree node
  (a function other than np.exp and np.log counts as the nodes it takes to write
  it with them and arithmetic: np.tanh(x) as 1 - 2/(1 + np.exp(2*x))). A model
  that only fits the training data will score poorly; the goal is a law
  that generalises across concentrations and gradients.

Rules:
  - Define N_PARAMS (int, at most {max_params}) and PARAM_INIT, a list
    of rough starting fitting values. Use p[i] for every constant you would want fitted.
  - Use only numpy (as np) and math, through plain functions such as np.exp,
    np.log, np.maximum (no np.random or other submodules). Fully vectorised, no
    loops over observations.
  - It must return finite values for every input in the stated ranges, including 0.
  - Signature exactly: def {signature}:
  - You may transform the inputs in any way you can justify.'''

SEED_COSTHETA = '''\
N_PARAMS = 1
PARAM_INIT = [10.0]

def model(C, G, p):
    # expected cos(theta) proportional to the relative gradient (0 where there is no gradient)
    costheta = p[0] * np.where(G > 0, G / np.maximum(C, 1e-12), 0.0)
    return costheta'''
SEED_CEI_2 = '''\
N_PARAMS = 1
PARAM_INIT = [0.5]

def model(C, G, p):
    # expected CEI: a ceiling p[0], lowered at small gradients and at small concentrations
    cei = p[0] * (1 - (0.001 / G + 0.1 / C))
    return cei'''

SEED_CEI_3 = '''\
N_PARAMS = 2
PARAM_INIT = [0.5, 1.0]

def model(C, G, p):
    # expected CEI: rises with the gradient and levels off at a ceiling p[0]
    cei = p[0] * (1 - np.exp(-G / p[1]))
    return cei'''

SETTINGS = {
    "data": ChemoTrack(root="${CHEMOTRACK_RETRACKED:-~/ChemoTrack_retracked}", x_range_um=(50.0, 950.0),
                       segmentation="authors", per_condition="equal", time_col=5, resample=True),
    "only_with_predecessor": False,
    "inputs": {"C": "C", "G": "G"},
    "likelihood": MeanCEI(),
    "system_prompt": SYSTEM,
    "prompt": PROMPT_CEI,
    "seeds": [SEED_CEI],
    "llm": LLM(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0,
               max_tokens=4000, timeout_s=600),
    "iterations": 30,
    "batch": 8,
    "islands": 1,
    "island_size": 20,
    "reset_every": 0,
    "examples_per_prompt": 3,
    "max_params": 8,
    "diagnostics_by": "concentration",
    "diagnostics_bins": 4,
    "diagnostics_rows_per_bin": 2000,
    "split_unit": "bridge",
    "split_block_um": 100.0,
    "split_block_origin_um": 0.0,
    "test_fraction": 0.25,
    "val_fraction": 0.25,
    "split_seeds": SPLIT_SEEDS,
    "fit_rows_per_condition": None,
    "val_rows_per_condition": None,
    "bin_rows_um": 100.0,
    "bin_weights": "equal",
    "op_costs": expansion_costs(),
    "param_weight": 1e-4,
    "complexity_weight": 1e-5,
}

TARGETS = {
    "cei": {},
    "speed": {"likelihood": Speed(shape=4.0, reference=REFERENCE_SPEED), "prompt": PROMPT_SPEED, "seeds": [SEED_SPEED]},
    "costheta": {"likelihood": MeanCosTheta(), "prompt": PROMPT_COSTHETA, "seeds": [SEED_COSTHETA]},
}
SCALES = {"0_01": 0.01, "0_1": 0.1, "10": 10.0, "20": 20.0, "30": 30.0, "100": 100.0}


def from_prompt_file(path: Path) -> dict:
    """A CEI variant from a prompt file: sections headed '=== NAME ===' (see prompts/)."""
    parts = re.split(r"^=== (.+?) ===\n", path.read_text(), flags=re.M)[1:]
    s = {name: text.strip("\n") for name, text in zip(parts[::2], parts[1::2])}
    likelihood = MeanCEI()
    likelihood.outputs = (s["OUTPUT NAME"].strip(),)
    likelihood.observable = s["OBSERVABLE"].strip()
    likelihood.sign_note = s["SIGN NOTE"]
    return {"system_prompt": s["SYSTEM"], "prompt": s["TASK"], "example_template": s["PREVIOUS MODEL"],
            "diagnostics_template": s["FEEDBACK"], "ask": s["ASK"], "seeds": [s["SEED"]], "likelihood": likelihood,
            "max_params": int(s["MAX PARAMS"])}


VARIANTS = {
    **TARGETS,
    **{f"{t}_x{name}": {**TARGETS[t], "param_weight": 1e-4 * f, "complexity_weight": 1e-5 * f}
       for t in ("cei", "speed") for name, f in SCALES.items()},
    "cei_seed2": {"seeds": [SEED_CEI_2]},
    "cei_seed3": {"seeds": [SEED_CEI_3]},
    **{f"cei_{p.stem}": from_prompt_file(p) for p in sorted((Path(__file__).parent / "prompts").glob("*.txt"))},
}

if __name__ == "__main__":
    variants_cli(VARIANTS, SETTINGS, repeats=REPEATS)
