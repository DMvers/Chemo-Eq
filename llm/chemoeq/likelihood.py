"""Distributions a candidate program can parameterise.

A likelihood says what the program returns (`outputs`), turns those values into
valid parameters (`prepare`), which measurement of a segment it describes
(`target`), and gives the log density of that measurement (`logpdf`). For
diagnostics and figures it also gives one number per segment, observed
(`observed`) and expected under the model (`mean_cos`): cos(theta) for the
direction families, the CEI for `CEI`. Scores are in bits per segment gained over
the family's no-information case (`log_null`: a uniformly random direction, a CEI
uniform on [-1, 1], or, for `Speed`, the same speed for every segment).

To try another family, write a class with the same members (or subclass one) and
set `likelihood=` in the experiment file.
"""
from __future__ import annotations

import math

import numpy as np

LOG2PI = math.log(2 * math.pi)


class CEI:
    """The chemotactic efficiency index of a segment, CEI = distance moved along +x /
    path length, in [-1, 1], with density

        f(cei) = kappa * exp(kappa * cei) / (2 * sinh(kappa)).

    The program returns kappa alone: 0 is no steering (CEI uniform on [-1, 1]),
    positive is up the gradient, negative down; |kappa| is capped at `kappa_max`.
    E[CEI] = coth(kappa) - 1/kappa. The likelihood depends on the data only through
    the mean CEI, so fitting it matches the mean CEI the program predicts to the
    observed one."""
    name = "cei"
    outputs = ("kappa",)
    observable = "CEI"
    symbol = "CEI"
    footnote = "mean CEI = coth κ − 1/κ, the expected distance moved along +x per unit of path."
    log_null = -math.log(2)                   # log density of a CEI uniform on [-1, 1]
    sign_note = "; positive = net movement towards +x"
    number_format = "+.3f"
    zero_based = True

    def __init__(self, kappa_max: float = 200.0):
        self.kappa_max = kappa_max

    def prepare(self, out, n):
        kappa = np.broadcast_to(np.asarray(out[0], float), (n,)).copy()
        return (np.clip(kappa, -self.kappa_max, self.kappa_max),)

    def target(self, X):
        if "cei" in X:
            return np.asarray(X["cei"], float)
        return np.asarray(X["straightness"], float) * np.cos(np.asarray(X["theta"], float))

    observed = target

    def logpdf(self, cei, kappa):
        a = np.abs(kappa)
        small = a < 1e-3
        b = np.where(small, 1.0, a)
        # log(sinh(a) / a), stable for large a; series near 0
        log_sinhc = np.where(small, a * a / 6.0, b + np.log1p(-np.exp(-2.0 * b)) - np.log(2.0 * b))
        return kappa * cei - math.log(2) - log_sinhc

    def mean_cos(self, kappa):
        """E[CEI] (the name is the one the direction families use for their observable)."""
        small = np.abs(kappa) < 1e-3
        k = np.where(small, 1.0, kappa)
        return np.where(small, kappa / 3.0, 1.0 / np.tanh(k) - 1.0 / k)

    def __repr__(self):
        return f"CEI(kappa_max={self.kappa_max})"


def langevin(k):
    """coth(k) - 1/k, the mean of the CEI density with steering strength k (k / 3 near 0)."""
    k = np.asarray(k, float)
    small = np.abs(k) < 1e-3
    s = np.where(small, 1.0, k)
    return np.where(small, k / 3.0, 1.0 / np.tanh(s) - 1.0 / s)


def inverse_langevin(m, iterations: int = 4):
    """The k with coth(k) - 1/k = m, for |m| < 1: Cohen's approximation m (3 - m^2) / (1 - m^2)
    (within 5%), then Newton steps on |m| (slope 1/k^2 - 1/sinh^2 k), which make it exact to
    about 1e-12."""
    m = np.asarray(m, float)
    a = np.abs(m)
    k = a * (3.0 - a * a) / (1.0 - a * a)
    for _ in range(iterations):
        small = k < 1e-3
        s = np.where(small, 1.0, k)
        e = np.exp(-2.0 * s)
        slope = np.where(small, 1.0 / 3.0, 1.0 / (s * s) - 4.0 * e / (1.0 - e) ** 2)
        k = np.maximum(k - (langevin(k) - a) / slope, 0.0)
    return np.sign(m) * k


class MeanCEI(CEI):
    """The same CEI density as `CEI`, but the program returns the expected CEI of a segment
    itself, from -1 to 1, rather than the steering strength kappa. The density with that
    mean, f(cei) = kappa exp(kappa cei) / (2 sinh kappa) with coth(kappa) - 1/kappa = the
    returned value, scores it, so scores are in the same bits as `CEI`'s. Values beyond
    what kappa_max allows (about +-0.995) are clipped to it."""
    name = "mean_cei"
    outputs = ("cei",)

    def prepare(self, out, n):
        cap = float(langevin(self.kappa_max))
        m = np.clip(np.broadcast_to(np.asarray(out[0], float), (n,)), -cap, cap)
        return (inverse_langevin(m),)

    def __repr__(self):
        return f"MeanCEI(kappa_max={self.kappa_max})"


class MeanCosTheta(MeanCEI):
    """The direction of a segment alone, cos(theta) (theta: the direction of its net displacement; +1 straight up
    the gradient, 0 across it, -1 straight down), scored exactly as MeanCEI scores the CEI (the CEI is
    straightness * cos theta): the program returns the expected cos theta and the maximum-entropy density on
    [-1, 1] with that mean scores it, in bits per segment over an expected cos theta of 0 everywhere."""
    name = "mean_costheta"
    outputs = ("costheta",)
    observable = "cos theta"
    symbol = "cos θ"
    footnote = "cos θ: the direction of a segment's net displacement relative to +x."

    def target(self, X):
        if "costheta" in X:
            return np.asarray(X["costheta"], float)
        return np.cos(np.asarray(X["theta"], float))

    observed = target

    def __repr__(self):
        return f"MeanCosTheta(kappa_max={self.kappa_max})"


class Speed:
    """The speed of a segment, its path length divided by its duration (um/min), as a
    gamma distribution with a fixed shape around the mean the program returns:

        f(s) = Gamma(s; shape, scale = speed / shape).

    The program returns the expected speed alone; it must be positive (values outside
    `min_speed`..`max_speed` are clipped). With the shape fixed, the likelihood depends
    on the data only through the mean speed, so fitting it matches the mean speed the
    program predicts to the observed one. The log density is taken relative to a gamma
    with the same shape and the mean `reference` for every segment, so scores are bits
    per segment gained over predicting that one speed everywhere."""
    name = "speed"
    outputs = ("speed",)
    observable = "speed (um/min)"
    symbol = "speed (µm/min)"
    footnote = "speed = path length of a segment / its duration."
    log_null = 0.0                            # the log density is already relative to the reference speed
    sign_note = ""
    number_format = ".2f"
    zero_based = False

    def __init__(self, shape: float = 4.0, reference: float = 10.4, min_speed: float = 0.05, max_speed: float = 1000.0):
        self.shape, self.reference, self.min_speed, self.max_speed = shape, reference, min_speed, max_speed

    def prepare(self, out, n):
        m = np.broadcast_to(np.asarray(out[0], float), (n,)).copy()
        return (np.clip(m, self.min_speed, self.max_speed),)

    def target(self, X):
        return np.asarray(X["speed_obs"] if "speed_obs" in X else X["speed"], float)

    observed = target

    def logpdf(self, s, m):
        # log Gamma(s; k, m/k) - log Gamma(s; k, m0/k): the terms in s alone cancel
        return self.shape * (s * (1.0 / self.reference - 1.0 / m) - np.log(m / self.reference))

    def mean_cos(self, m):
        """E[speed] (the name is the one the direction families use for their observable)."""
        return m

    def __repr__(self):
        return f"Speed(shape={self.shape}, reference={self.reference})"


def gain_bits(mean_logpdf: float, likelihood=None) -> float:
    """Information gained per segment versus the family's no-information case
    (a uniformly random direction unless the likelihood says otherwise)."""
    null = getattr(likelihood, "log_null", -LOG2PI)
    return (mean_logpdf - null) / math.log(2)
