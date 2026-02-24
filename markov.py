# markov.py
# 中文说明：生产率x的Tauchen离散化与β三点Markov链构造。

import numpy as np
from math import erf, sqrt


def _norm_cdf(x: float) -> float:
    """标准正态CDF（无scipy版本）。"""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def tauchen(rho: float, sigma: float, N: int, m: float = 3.0):
    """
    Tauchen (1986) 离散化：log x 服从 AR(1)。
    返回：x_grid（水平变量，已指数化）、转移矩阵P（行：当前状态，列：下一期）。
    """
    z_std = sigma / sqrt(1 - rho ** 2)
    z_max = m * z_std
    z_min = -z_max
    z = np.linspace(z_min, z_max, N)
    step = (z_max - z_min) / (N - 1)

    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j == 0:
                P[i, j] = _norm_cdf((z[0] - rho * z[i] + step / 2) / sigma)
            elif j == N - 1:
                P[i, j] = 1 - _norm_cdf((z[-1] - rho * z[i] - step / 2) / sigma)
            else:
                upper = (z[j] - rho * z[i] + step / 2) / sigma
                lower = (z[j] - rho * z[i] - step / 2) / sigma
                P[i, j] = _norm_cdf(upper) - _norm_cdf(lower)

    x_grid = np.exp(z)
    return x_grid, P


def beta_markov_three_point(pi_stay: float = 0.995):
    """
    3点β转移矩阵，满足：
    - 每个状态以 pi_stay 持续不变；
    - 若发生转换，只向相邻状态移动（边界状态只能向中间移动）。
    """
    P = np.zeros((3, 3))
    switch = 1.0 - pi_stay

    P[0, 0] = pi_stay
    P[0, 1] = switch

    P[1, 1] = pi_stay
    P[1, 0] = switch / 2.0
    P[1, 2] = switch / 2.0

    P[2, 2] = pi_stay
    P[2, 1] = switch

    return P
