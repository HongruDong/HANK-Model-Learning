# equilibrium.py
# 中文说明：
# 给定猜测序列 X = {r_k, Pi, w_h, lambda, d}_{t=0..T-1}，构造误差向量 E(X)=0。

import numpy as np
from typing import Dict, Tuple

from household import solve_transition


def taylor_rule(Pi: np.ndarray, phi_Pi: float, i_bar: float, Pi_bar: float) -> np.ndarray:
    T = len(Pi)
    i = np.zeros(T)
    i[0] = i_bar
    for t in range(1, T):
        i[t] = (1.0 + i_bar) * (Pi[t - 1] / Pi_bar) ** phi_Pi - 1.0
    return i


def fisher_real_rate(i: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    return (1.0 + i) / Pi - 1.0


def qk_from_no_arbitrage(r: np.ndarray, r_k: np.ndarray, qk_ss: float = 1.0) -> np.ndarray:
    T = len(r)
    qk = np.zeros(T + 1)
    qk[T] = qk_ss
    for t in range(T - 1, -1, -1):
        r_next = r[t] if t == T - 1 else r[t + 1]
        rk_next = r_k[t] if t == T - 1 else r_k[t + 1]
        qk[t] = (qk[t + 1] + rk_next) / (1.0 + r_next)
    return qk[:T]


def nkpc_implied_M(Pi: np.ndarray, Y: np.ndarray, r: np.ndarray, eps: float, Theta: float, Pi_bar: float) -> np.ndarray:
    T = len(Pi)
    M = np.zeros(T)
    for t in range(T):
        Pi_t = Pi[t]
        r_next = r[t] if t == T - 1 else r[t + 1]
        Pi_next = Pi[t] if t == T - 1 else Pi[t + 1]
        Y_ratio = 1.0 if t == T - 1 else (Y[t + 1] / max(1e-12, Y[t]))
        lhs = (Pi_t - Pi_bar) * Pi_t + (eps - 1.0) / Theta
        fwd = (1.0 / (1.0 + r_next)) * (Pi_next - Pi_bar) * Pi_next * Y_ratio
        M[t] = (Theta / eps) * (lhs - fwd)
    return M


def wage_from_unit_cost(M: np.ndarray, r_cost: np.ndarray, A_g: np.ndarray, alpha: float) -> np.ndarray:
    T = len(M)
    w = np.zeros(T)
    for t in range(T):
        u = r_cost[t] / (1.0 - alpha)
        denom = (u ** (1.0 - alpha)) if u > 1e-14 else 1e-14
        inside = max(1e-14, M[t] * A_g[t] / denom)
        w[t] = alpha * (inside ** (1.0 / alpha))
    return w


def wage_inflation(w: np.ndarray, Pi: np.ndarray, w_ss: float) -> np.ndarray:
    T = len(w)
    Pi_w = np.zeros(T)
    w_lag = w_ss
    for t in range(T):
        Pi_w[t] = Pi[t] * (w[t] / max(1e-14, w_lag))
        w_lag = w[t]
    return Pi_w


def wh_from_wage_pc(w: np.ndarray, N: np.ndarray, r: np.ndarray, Pi_w: np.ndarray,
                    eps_w: float, Theta_w: float, Pi_bar: float) -> np.ndarray:
    T = len(w)
    wh = np.zeros(T)
    for t in range(T):
        r_next = r[t] if t == T - 1 else r[t + 1]
        Pi_w_next = Pi_w[t] if t == T - 1 else Pi_w[t + 1]
        N_ratio = 1.0 if t == T - 1 else (N[t + 1] / max(1e-14, N[t]))
        lhs = (Pi_w[t] - Pi_bar) * Pi_w[t] + ((eps_w - 1.0) / Theta_w) * w[t]
        fwd = (1.0 / (1.0 + r_next)) * (Pi_w_next - Pi_bar) * Pi_w_next * N_ratio
        wh[t] = (Theta_w / eps_w) * (lhs - fwd)
    return wh


def capital_foc_rk(M: np.ndarray, Y: np.ndarray, K: np.ndarray, qk: np.ndarray, delta: float, alpha: float) -> np.ndarray:
    T = len(M)
    rk_hat = np.zeros(T)
    for t in range(T):
        mpk_term = (1.0 - alpha) * M[t] * Y[t] / max(1e-14, K[t])
        rk_hat[t] = mpk_term - delta * qk[t]
    return rk_hat


def qk_from_capital_producer(K: np.ndarray, delta: float, phi_k: float) -> np.ndarray:
    T = len(K) - 1
    qhat = np.zeros(T)
    for t in range(T):
        Delta = K[t + 1] - (1.0 - delta) * K[t]
        qhat[t] = 1.0 + phi_k * (Delta / max(1e-14, K[t]) - delta)
    return qhat


def profits_and_dividends(Y, w, wh, N, rk, qk, K, Pi, Pi_w, params, fixed_costs):
    delta = params.delta()
    Theta = params.Theta
    Theta_w = params.Theta_w
    phi_k = params.phi_k
    Pi_bar = params.Pi_bar

    T = len(Y)
    d_hat = np.zeros(T)

    Phi, Phi_w = fixed_costs

    for t in range(T):
        price_adj = (Theta / 2.0) * ((Pi[t] - Pi_bar) ** 2) * Y[t]
        wage_adj = (Theta_w / 2.0) * ((Pi_w[t] - Pi_bar) ** 2) * N[t]
        r_cost = rk[t] + delta * qk[t]
        profit_int = Y[t] - w[t] * N[t] - r_cost * K[t] - price_adj - Phi
        profit_union = (w[t] - wh[t]) * N[t] - wage_adj - Phi_w

        DeltaK = K[t + 1] - (1.0 - delta) * K[t]
        invest_cost = DeltaK + (phi_k / 2.0) * ((DeltaK / max(1e-14, K[t]) - delta) ** 2) * K[t]
        profit_cap = qk[t] * DeltaK - invest_cost

        d_hat[t] = profit_int + profit_union + profit_cap

    return d_hat


def equilibrium_residual(X: np.ndarray, params, grids, markov, ss, policy) -> Tuple[np.ndarray, Dict]:
    T = params.T
    a_grid = grids["a"]
    x_grid = grids["x"]
    beta_grid = grids["beta"]
    Px = markov["Px"]
    Pbeta = markov["Pbeta"]

    rk = X[0:T]
    Pi = X[T:2*T]
    wh = X[2*T:3*T]
    lam = X[3*T:4*T]
    div = X[4*T:5*T]

    Gc = policy["Gc"]
    Ig = policy["Ig"]
    transfer = policy["transfer"]

    G_total = Gc + Ig
    gamma = params.gamma_ss + params.phi_progress * (G_total - params.G_ss)

    i_bar = (1.0 + ss["r_ss"]) * params.Pi_bar - 1.0
    i = taylor_rule(Pi, params.phi_Pi, i_bar=i_bar, Pi_bar=params.Pi_bar)
    r = fisher_real_rate(i, Pi)

    qk = qk_from_no_arbitrage(r, rk, qk_ss=ss["qk_ss"])

    seq = dict(w_h=wh, r=r, lam=lam, gamma=gamma, transfer=transfer, div=div)
    hh = solve_transition(
        a_grid, x_grid, beta_grid, Px, Pbeta, params, ss["household_ss"], seq,
        progress_cb=policy.get("transition_progress_cb")
    )

    A, N, C, Taxes = hh["A"], hh["N"], hh["C"], hh["Taxes"]

    D = np.zeros(T + 1)
    D[0] = params.D_ss
    # 统一口径：债务更新规则使用“总税收 Taxes（与家庭块一致）”而不是仅资本税近似。
    Taxes_ss = float(ss["household_ss"]["Taxes"])
    F_ss = params.G_ss + (1.0 + ss["r_ss"]) * params.D_ss + params.Transfers_ss - Taxes_ss
    for t in range(T):
        Ft = G_total[t] + (1.0 + r[t]) * D[t] + transfer[t] - Taxes[t]
        D[t + 1] = params.D_ss + params.theta_deficit * (Ft - F_ss)

    K = np.zeros(T + 1)
    K[0] = (A[0] - D[0]) / max(1e-14, ss["qk_ss"])
    for t in range(T):
        K[t + 1] = (A[t + 1] - D[t + 1]) / max(1e-14, qk[t])

    Kg = np.zeros(T + 1)
    Kg[0] = params.Kg_ss
    for t in range(T):
        Kg[t + 1] = (1.0 - params.delta_g) * Kg[t] + Ig[t]

    A_g = Kg[:T] ** params.psi_g
    alpha = params.alpha
    Y = A_g * (np.maximum(1e-14, K[:T]) ** (1.0 - alpha)) * (np.maximum(1e-14, N) ** alpha)

    M = nkpc_implied_M(Pi, Y, r, params.eps, params.Theta, params.Pi_bar)
    r_cost = rk + params.delta() * qk
    w = wage_from_unit_cost(M, r_cost, A_g, alpha)

    Pi_w = wage_inflation(w, Pi, w_ss=ss["w_ss"])
    wh_hat = wh_from_wage_pc(w, N, r, Pi_w, params.eps_w, params.Theta_w, params.Pi_bar)

    rk_hat = capital_foc_rk(M, Y, K[:T], qk, params.delta(), alpha)
    qk_hat = qk_from_capital_producer(K, params.delta(), params.phi_k)

    EG = (G_total + (1.0 + r) * D[:T] + transfer) - D[1:] - Taxes

    d_hat = profits_and_dividends(
        Y=Y,
        w=w,
        wh=wh,
        N=N,
        rk=rk,
        qk=qk,
        K=K,
        Pi=Pi,
        Pi_w=Pi_w,
        params=params,
        fixed_costs=(ss["Phi"], ss["Phi_w"]),
    )

    Er = rk - rk_hat
    Eq = qk - qk_hat
    Ew = wh - wh_hat
    Ed = div - d_hat

    E = np.concatenate([Er, Eq, Ew, EG, Ed], axis=0)

    goods_resid = np.zeros(T)
    for t in range(T):
        DeltaK = K[t + 1] - (1.0 - params.delta()) * K[t]
        invest_cost = DeltaK + (params.phi_k / 2.0) * ((DeltaK / max(1e-14, K[t]) - params.delta()) ** 2) * K[t]
        price_adj = (params.Theta / 2.0) * ((Pi[t] - params.Pi_bar) ** 2) * Y[t]
        wage_adj = (params.Theta_w / 2.0) * ((Pi_w[t] - params.Pi_bar) ** 2) * N[t]
        goods_resid[t] = Y[t] - (C[t] + G_total[t] + invest_cost + price_adj + wage_adj + ss["Phi"] + ss["Phi_w"])

    extras = dict(
        r=r, i=i, Pi=Pi, qk=qk, D=D, K=K, Kg=Kg, Y=Y, M=M, w=w, Pi_w=Pi_w, hh=hh, goods_resid=goods_resid,
        rk_hat=rk_hat, wh_hat=wh_hat, d_hat=d_hat,
        Er=Er, Eq=Eq, Ew=Ew, EG=EG, Ed=Ed,
    )
    return E, extras
