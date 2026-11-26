"""The LLM search's likelihoods (chemoeq.likelihood MeanCEI, MeanCosTheta and Speed) in NumPy and as PySR losses in
Julia.

CEI and cos theta: a model returns the expected value m of a segment, scored with the density on [-1, 1] that has that
mean, f(y) = k exp(k y) / (2 sinh k), where coth(k) - 1/k = m. For a bin (the mean y of its segments) this is the log
density of y, so a fit matches the predicted mean to the observed one. Gains are bits per bin over 0 everywhere.

Speed: a model returns the expected speed m (um/min), clipped to SPEED_MIN..SPEED_MAX, scored with a gamma of shape 4
around m relative to the same gamma around SPEED_REFERENCE (16.4 um/min, the mean speed of the segments):
4 (s (1/ref - 1/m) - log(m / ref)). Gains are bits per bin over predicting the reference everywhere.
"""
import math

import numpy as np

KAPPA_MAX = 200.0
CAP = 1.0 / math.tanh(KAPPA_MAX) - 1.0 / KAPPA_MAX        # 0.995: the largest |mean| a CEI model can return
SPEED_SHAPE, SPEED_REFERENCE, SPEED_MIN, SPEED_MAX = 4.0, 16.4, 0.05, 1000.0


def langevin(k):
    """coth(k) - 1/k, the mean at steering strength k (k / 3 near 0)."""
    k = np.asarray(k, float)
    small = np.abs(k) < 1e-3
    s = np.where(small, 1.0, k)
    return np.where(small, k / 3.0, 1.0 / np.tanh(s) - 1.0 / s)


def inverse_langevin(m, iterations: int = 4):
    """The k with coth(k) - 1/k = m (Cohen's approximation, then Newton steps), as chemoeq computes it."""
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


def cei_logpdf(y, m):
    """Log density of the (mean) CEI or cos theta y under the expected value m (clipped to +-CAP)."""
    with np.errstate(all="ignore"):
        m = np.clip(np.asarray(m, float), -CAP, CAP)
        k = inverse_langevin(m)
        a = np.abs(k)
        small = a < 1e-3
        b = np.where(small, 1.0, a)
        log_sinhc = np.where(small, a * a / 6.0, b + np.log1p(-np.exp(-2.0 * b)) - np.log(2.0 * b))
        return k * np.asarray(y, float) - math.log(2) - log_sinhc


def speed_logpdf(s, m):
    with np.errstate(all="ignore"):
        m = np.clip(np.asarray(m, float), SPEED_MIN, SPEED_MAX)
        return SPEED_SHAPE * (np.asarray(s, float) * (1.0 / SPEED_REFERENCE - 1.0 / m) - np.log(m / SPEED_REFERENCE))


LOGPDF = {"cei": cei_logpdf, "costheta": cei_logpdf, "speed": speed_logpdf}
LOG_NULL = {"cei": -math.log(2), "costheta": -math.log(2), "speed": 0.0}
LOSS_OFFSET = {"cei": 1.0, "costheta": 1.0, "speed": 2.0}      # added to the Julia losses to keep them positive


def gain_bits(y, m, target: str) -> float:
    """Mean gain in bits per bin over the target's null model; -inf if m is not finite."""
    m = np.broadcast_to(np.asarray(m, float), np.shape(y))
    if not np.all(np.isfinite(m)):
        return -math.inf
    return float((np.mean(LOGPDF[target](y, m)) - LOG_NULL[target]) / math.log(2))


def pysr_loss(y, m, target: str) -> float:
    """The mean of the Julia loss, as PySR computes it: minus the gain in nats per bin, plus the offset."""
    lp = LOGPDF[target](y, np.broadcast_to(np.asarray(m, float), np.shape(y)))
    return float(-(np.mean(lp) - LOG_NULL[target]) + LOSS_OFFSET[target])


# The same losses for PySR, per bin: minus the log gain in nats, plus the offset (PySR's model selection takes logs of
# losses). The steps follow inverse_langevin() exactly.
CEI_LOSS = f"""
function cei_loss(m::T, y::T) where {{T}}
    cap = T({CAP!r})
    a = min(abs(m), cap)
    k = a * (3 - a * a) / (1 - a * a)
    for _ in 1:4
        small = k < T(1e-3)
        s = small ? one(T) : k
        e = exp(-2 * s)
        slope = small ? T(1) / 3 : 1 / (s * s) - 4 * e / (1 - e)^2
        L = small ? k / 3 : 1 / tanh(s) - 1 / s
        k = max(k - (L - a) / slope, zero(T))
    end
    k = m < 0 ? -k : k
    b = abs(k)
    log_sinhc = b < T(1e-3) ? b * b / 6 : b + log1p(-exp(-2 * b)) - log(2 * b)
    return log_sinhc - k * y + one(T)
end
cei_loss(m::T, y::T, w::T) where {{T}} = w * cei_loss(m, y)
"""

SPEED_LOSS = f"""
function speed_loss(m::T, s::T) where {{T}}
    mm = clamp(m, T({SPEED_MIN!r}), T({SPEED_MAX!r}))
    return T({SPEED_SHAPE!r}) * (log(mm / T({SPEED_REFERENCE!r})) - s * (1 / T({SPEED_REFERENCE!r}) - 1 / mm)) + T(2)
end
speed_loss(m::T, s::T, w::T) where {{T}} = w * speed_loss(m, s)
"""

JULIA_LOSS = {"cei": CEI_LOSS, "costheta": CEI_LOSS, "speed": SPEED_LOSS}
