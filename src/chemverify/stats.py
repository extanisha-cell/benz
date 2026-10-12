"""Small, dependency-free statistics: exact and Wilson intervals, and a cluster bootstrap."""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        log_term = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        total += math.exp(log_term)
    return min(1.0, total)


def _root_increasing(f: Callable[[float], float], iters: int = 80) -> float:
    """Root in [0, 1] of an increasing function, by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided (1 - alpha) interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    lower = 0.0 if k <= 0 else _root_increasing(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2)
    upper = 1.0 if k >= n else _root_increasing(lambda p: alpha / 2 - _binom_cdf(k, n, p))
    return (lower, upper)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper (1 - alpha) bound: 'with 95% confidence the true rate is below this'."""
    if n <= 0 or k >= n:
        return 1.0
    return _root_increasing(lambda p: alpha - _binom_cdf(k, n, p))


def n_for_zero_failures(rate: float, confidence: float = 0.95) -> int:
    """Failure-free trials needed before the one-sided upper bound drops below `rate`."""
    return math.ceil(math.log(1 - confidence) / math.log(1 - rate))


def ratio_bootstrap(
    clusters: Sequence[tuple[float, float]], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float | None, float | None]:
    """CI for sum(num)/sum(den), resampling whole clusters (pairs): runs on one pair are correlated."""
    usable = [c for c in clusters if c[1] > 0]
    if not usable:
        return (None, None)
    rng = random.Random(seed)
    m = len(usable)
    stats = []
    for _ in range(n_boot):
        num = den = 0.0
        for _ in range(m):
            c = usable[rng.randrange(m)]
            num += c[0]
            den += c[1]
        stats.append(num / den)
    stats.sort()
    return (stats[int(alpha / 2 * n_boot)], stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])
