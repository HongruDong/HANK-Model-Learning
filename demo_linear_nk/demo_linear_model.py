# demo_linear_nk/demo_linear_model.py
# 中文说明：线性NK示意模型（不含完整HANK分布），用于快速画IRF。

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def solve_linear_nk(T=60, rbar=0.0086, kappa=0.035, phi_pi=1.5, psi_k=0.8, u=None, Kg=None):
    beta = 1.0 / (1.0 + rbar)
    if u is None:
        u = np.zeros(T)
    if Kg is None:
        Kg = np.zeros(T)

    n = 2 * T
    A = np.zeros((n, n))
    b = np.zeros(n)

    def idx_pi(t):
        return t

    def idx_x(t):
        return T + t

    for t in range(T):
        A[idx_pi(t), idx_pi(t)] = 1.0
        if t < T - 1:
            A[idx_pi(t), idx_pi(t + 1)] = -beta
        A[idx_pi(t), idx_x(t)] = -kappa
        b[idx_pi(t)] = kappa * psi_k * Kg[t]

        row = idx_x(t)
        A[row, idx_x(t)] = 1.0
        if t < T - 1:
            A[row, idx_x(t + 1)] = -1.0
            A[row, idx_pi(t + 1)] = 1.0
        A[row, idx_pi(t)] = phi_pi
        b[row] = rbar + u[t]

    sol = np.linalg.solve(A, b)
    pi = sol[:T]
    x = sol[T:]
    p = np.cumsum(pi)
    return dict(pi=pi, x=x, p=p)


def main():
    T = 60
    rbar = 0.0086
    phi_pi = 1.5
    kappa = 0.035

    rho_I = 0.9
    Ig0 = 0.01
    Ig = Ig0 * rho_I ** np.arange(T)

    delta_g = 0.02
    Kg = np.zeros(T)
    for t in range(1, T):
        Kg[t] = (1 - delta_g) * Kg[t - 1] + Ig[t - 1]

    phi_b = 0.3
    b = np.zeros(T)
    for t in range(1, T):
        b[t] = (1 + rbar) * b[t - 1] + Ig[t - 1] - phi_b * b[t - 1]

    chi_b = 0.6
    u = Ig - chi_b * b

    res = solve_linear_nk(T=T, rbar=rbar, kappa=kappa, phi_pi=phi_pi, psi_k=0.8, u=u, Kg=Kg)
    res0 = solve_linear_nk(T=T, rbar=rbar, kappa=kappa, phi_pi=phi_pi, psi_k=0.8, u=u, Kg=np.zeros(T))

    t = np.arange(T)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(t, res["pi"] * 100)
    plt.plot(t, res0["pi"] * 100)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Quarters")
    plt.ylabel("Inflation deviation (pp)")
    plt.title("Demo IRF: inflation")
    plt.legend(["With public capital", "No public capital"])
    fig.tight_layout()
    fig.savefig("irf_pi_demo.png", dpi=150)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(t, res["p"] * 100)
    plt.plot(t, res0["p"] * 100)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Quarters")
    plt.ylabel("Price level (cum. pp)")
    plt.title("Demo IRF: price level")
    plt.legend(["With public capital", "No public capital"])
    fig.tight_layout()
    fig.savefig("irf_price_demo.png", dpi=150)

    pd.DataFrame(
        {
            "t": t,
            "pi_with": res["pi"],
            "pi_noK": res0["pi"],
            "p_with": res["p"],
            "p_noK": res0["p"],
        }
    ).to_csv("demo_irf.csv", index=False)


if __name__ == "__main__":
    main()
