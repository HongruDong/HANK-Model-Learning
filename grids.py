# grids.py
# 中文说明：网格构造与Markov链稳态分布工具。

import numpy as np


def make_asset_grid(Na: int, a_max: float, curv: float) -> np.ndarray:
    """
    非线性加密资产网格：a_i = (i/(Na-1))^curv * a_max
    curv>1 在低资产区更密集，有助于刻画借贷约束附近的高MPC。
    """
    x = np.linspace(0.0, 1.0, Na)
    return (x ** curv) * a_max


def stationary_dist_markov(P: np.ndarray, tol: float = 1e-14, maxit: int = 200000) -> np.ndarray:
    """
    求离散Markov链P的稳态分布π，满足 π = π P。
    迭代法：π_{n+1} = π_n P。
    """
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(maxit):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new
    return pi
