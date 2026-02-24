# household.py
# 中文说明：
# 1) 稳态：给定(w_h, r, lambda, gamma, transfer, dividends) 解无限期VFI + 分布迭代；
# 2) 过渡：给定序列 {w_h_t, r_t, lambda_t, gamma_t, transfer_t, d_t}，
#    用“终端V=稳态V”的条件进行向后递推，再向前推进分布与聚合量。

import time
import numpy as np
from typing import Dict, Tuple, Optional, Callable

BIG_NEG = -1e30


def labor_tax_rate(y_l: float, lam: float, gamma: float) -> float:
    """tau_l(y) = 1 - lam * y^{-gamma}."""
    if y_l <= 1e-14:
        return 0.0
    return 1.0 - lam * (y_l ** (-gamma))


def labor_tax_rate_hat(y_l: float, lam: float, gamma_ss: float, gamma_t: float) -> float:
    tau_ss = labor_tax_rate(y_l, lam, gamma_ss)
    tau_t = labor_tax_rate(y_l, lam, gamma_t)
    return max(tau_ss, tau_t)


def total_tax(wxh: float, ra: float, tau_k: float, lam: float, gamma_ss: float, gamma_t: float) -> float:
    tau_l = labor_tax_rate_hat(wxh, lam, gamma_ss, gamma_t)
    return tau_k * ra + tau_l * wxh


def logsumexp2(a: np.ndarray, b: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    m = np.maximum(a, b)
    ea = np.exp((a - m) / scale)
    eb = np.exp((b - m) / scale)
    s = ea + eb
    V = m + scale * np.log(s)
    p = eb / s
    return V, p


def expected_value_next(Vnext: np.ndarray, Px: np.ndarray, Pbeta: np.ndarray) -> np.ndarray:
    Na, _, _ = Vnext.shape
    EV = np.zeros_like(Vnext)
    for i in range(Na):
        EV[i] = Px @ Vnext[i] @ Pbeta.T
    return EV




def _howard_policy_evaluation(V, pol_a_idx0, pol_a_idx1, c0, c1, beta_grid, Px, Pbeta, params, n_iter: int):
    """在固定策略下进行Howard策略评估，加速值函数收敛。"""
    hbar = params.hbar
    Vcur = V.copy()
    pwork = None
    for _ in range(n_iter):
        EV = expected_value_next(Vcur, Px, Pbeta)
        Na, Nx, Nb = Vcur.shape
        Vh0 = np.full_like(Vcur, BIG_NEG)
        Vh1 = np.full_like(Vcur, BIG_NEG)
        for ia in range(Na):
            for ix in range(Nx):
                for ib, b in enumerate(beta_grid):
                    ip0 = pol_a_idx0[ia, ix, ib]
                    ip1 = pol_a_idx1[ia, ix, ib]
                    Vh0[ia, ix, ib] = np.log(max(c0[ia, ix, ib], 1e-14)) + b * EV[ip0, ix, ib]
                    Vh1[ia, ix, ib] = np.log(max(c1[ia, ix, ib], 1e-14)) - params.B * hbar + b * EV[ip1, ix, ib]
        Vcur, pwork = logsumexp2(Vh0, Vh1, params.varrho)
    return Vcur, pwork


def solve_steady_state(
    a_grid: np.ndarray,
    x_grid: np.ndarray,
    beta_grid: np.ndarray,
    Px: np.ndarray,
    Pbeta: np.ndarray,
    params,
    w_h: float,
    r: float,
    lam: float,
    gamma_ss: float,
    transfer: float,
    d_total: float = 0.0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    Na, Nx, Nb = len(a_grid), len(x_grid), len(beta_grid)
    hbar = params.hbar

    from grids import stationary_dist_markov

    pi_x = stationary_dist_markov(Px)
    Ex = float(np.sum(pi_x * x_grid))

    dbar = d_total / Ex if Ex > 0 else 0.0
    div_x = dbar * x_grid

    V = np.zeros((Na, Nx, Nb))
    pol_a_idx0 = np.zeros((Na, Nx, Nb), dtype=np.int32)
    pol_a_idx1 = np.zeros((Na, Nx, Nb), dtype=np.int32)
    c0 = np.zeros((Na, Nx, Nb))
    c1 = np.zeros((Na, Nx, Nb))
    tax0 = np.zeros((Na, Nx, Nb))
    tax1 = np.zeros((Na, Nx, Nb))
    pwork = np.zeros((Na, Nx, Nb))

    t0 = time.time()
    prev_pol0 = None
    prev_pol1 = None
    stable_hits = 0
    for it in range(params.maxit_vfi):
        EV = expected_value_next(V, Px, Pbeta)
        Vh0 = np.full((Na, Nx, Nb), BIG_NEG)
        Vh1 = np.full((Na, Nx, Nb), BIG_NEG)

        for ia, a in enumerate(a_grid):
            for ix, x in enumerate(x_grid):
                for ib, b in enumerate(beta_grid):
                    for hflag in (0, 1):
                        h = hbar if hflag == 1 else 0.0
                        labor_income = w_h * x * h
                        ra = r * a
                        T = total_tax(labor_income, ra, params.tau_k, lam, gamma_ss, gamma_ss)
                        res = labor_income + (1.0 + r) * a - T + transfer + div_x[ix]
                        best_val = BIG_NEG
                        best_ip = 0
                        best_c = 0.0
                        best_tax = T
                        for ip, ap in enumerate(a_grid):
                            c = res - ap
                            if c <= 1e-14:
                                continue
                            val = np.log(c) - params.B * h + b * EV[ip, ix, ib]
                            if val > best_val:
                                best_val = val
                                best_ip = ip
                                best_c = c
                                best_tax = T
                        if hflag == 0:
                            Vh0[ia, ix, ib] = best_val
                            pol_a_idx0[ia, ix, ib] = best_ip
                            c0[ia, ix, ib] = best_c
                            tax0[ia, ix, ib] = best_tax
                        else:
                            Vh1[ia, ix, ib] = best_val
                            pol_a_idx1[ia, ix, ib] = best_ip
                            c1[ia, ix, ib] = best_c
                            tax1[ia, ix, ib] = best_tax

        Vnew, p = logsumexp2(Vh0, Vh1, params.varrho)
        diff = np.max(np.abs(Vnew - V))
        diff_rel = diff / max(1.0, float(np.max(np.abs(V))))

        if prev_pol0 is None:
            pol_change = np.inf
        else:
            pol_change = max(
                int(np.max(np.abs(pol_a_idx0 - prev_pol0))),
                int(np.max(np.abs(pol_a_idx1 - prev_pol1))),
            )

        if (it + 1) >= params.minit_vfi_policy and pol_change <= params.tol_vfi_policy:
            stable_hits += 1
        else:
            stable_hits = 0

        V = Vnew
        pwork = p

        # 策略稳定后启用Howard策略评估，提升值函数收敛速度
        if stable_hits >= params.howard_start_hits and params.howard_iter > 0:
            V_before_howard = V.copy()
            V, pwork = _howard_policy_evaluation(
                V=V,
                pol_a_idx0=pol_a_idx0,
                pol_a_idx1=pol_a_idx1,
                c0=c0,
                c1=c1,
                beta_grid=beta_grid,
                Px=Px,
                Pbeta=Pbeta,
                params=params,
                n_iter=params.howard_iter,
            )
            diff = np.max(np.abs(V - V_before_howard))
            diff_rel = diff / max(1.0, float(np.max(np.abs(V_before_howard))))
            if progress_cb is not None:
                progress_cb(
                    f"[steady/VFI] Howard applied: n={params.howard_iter}, post_diff={diff:.3e}, stable_hits={stable_hits}"
                )

        if progress_cb is not None and (it == 0 or (it + 1) % params.progress_vfi_every == 0):
            pol_change_str = "inf" if np.isinf(pol_change) else str(pol_change)
            progress_cb(
                f"[steady/VFI] it={it+1}/{params.maxit_vfi}, diff={diff:.3e}, diff_rel={diff_rel:.3e}, "
                f"dpol={pol_change_str}, stable_hits={stable_hits}/{params.vfi_policy_stable_hits}, "
                f"policy_value_gate=({params.tol_vfi_policy_value:.1e},{params.tol_vfi_policy_value_rel:.1e}), elapsed={time.time()-t0:.1f}s"
            )

        if (diff < params.tol_vfi) or (diff_rel < params.tol_vfi_rel):
            if progress_cb is not None:
                progress_cb(
                    f"[steady/VFI] converged by value at it={it+1}, diff={diff:.3e}, diff_rel={diff_rel:.3e}"
                )
            break

        if stable_hits >= params.vfi_policy_stable_hits and (
            (diff <= params.tol_vfi_policy_value) or (diff_rel <= params.tol_vfi_policy_value_rel)
        ):
            if progress_cb is not None:
                progress_cb(
                    f"[steady/VFI] converged by policy+value at it={it+1}, "
                    f"dpol<={params.tol_vfi_policy} for {stable_hits} consecutive iterations, "
                    f"and (diff={diff:.3e}, diff_rel={diff_rel:.3e}) "
                    f"<= gates ({params.tol_vfi_policy_value:.3e}, {params.tol_vfi_policy_value_rel:.3e})"
                )
            break

        prev_pol0 = pol_a_idx0.copy()
        prev_pol1 = pol_a_idx1.copy()
    else:
        if progress_cb is not None:
            progress_cb(f"[steady/VFI] WARNING: reached maxit={params.maxit_vfi}, last diff={diff:.3e}")

    mu = np.ones((Na, Nx, Nb)) / (Na * Nx * Nb)
    t1 = time.time()
    for it in range(params.maxit_dist):
        mu_new = np.zeros_like(mu)
        for ia in range(Na):
            for ix in range(Nx):
                for ib in range(Nb):
                    mass = mu[ia, ix, ib]
                    if mass == 0:
                        continue
                    pw = pwork[ia, ix, ib]
                    ip0 = pol_a_idx0[ia, ix, ib]
                    ip1 = pol_a_idx1[ia, ix, ib]
                    for jx in range(Nx):
                        for jb in range(Nb):
                            prob = Px[ix, jx] * Pbeta[ib, jb]
                            mu_new[ip0, jx, jb] += mass * (1.0 - pw) * prob
                            mu_new[ip1, jx, jb] += mass * pw * prob
        dist_diff = np.max(np.abs(mu_new - mu))
        mu = mu_new

        if progress_cb is not None and (it == 0 or (it + 1) % params.progress_dist_every == 0):
            progress_cb(f"[steady/dist] it={it+1}/{params.maxit_dist}, diff={dist_diff:.3e}, elapsed={time.time()-t1:.1f}s")

        if dist_diff < params.tol_dist:
            if progress_cb is not None:
                progress_cb(f"[steady/dist] converged at it={it+1}, diff={dist_diff:.3e}")
            break
    else:
        if progress_cb is not None:
            progress_cb(f"[steady/dist] WARNING: reached maxit={params.maxit_dist}, last diff={dist_diff:.3e}")

    A = float(np.sum(mu * a_grid[:, None, None]))
    N_eff = float(np.sum(mu * (x_grid[None, :, None] * params.hbar) * pwork))
    C = float(np.sum(mu * ((1.0 - pwork) * c0 + pwork * c1)))
    Taxes = float(np.sum(mu * ((1.0 - pwork) * tax0 + pwork * tax1)))

    return dict(
        V=V,
        mu=mu,
        A=A,
        N=N_eff,
        C=C,
        Taxes=Taxes,
        Ex=Ex,
        pol_a_idx0=pol_a_idx0,
        pol_a_idx1=pol_a_idx1,
        c0=c0,
        c1=c1,
        tax0=tax0,
        tax1=tax1,
        pwork=pwork,
    )


def solve_transition(
    a_grid: np.ndarray,
    x_grid: np.ndarray,
    beta_grid: np.ndarray,
    Px: np.ndarray,
    Pbeta: np.ndarray,
    params,
    ss: Dict,
    seq: Dict,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    T = params.T
    Na, Nx, Nb = len(a_grid), len(x_grid), len(beta_grid)

    V_T = ss["V"]
    mu0 = ss["mu"]
    Ex = ss["Ex"]

    pol0 = np.zeros((T, Na, Nx, Nb), dtype=np.int32)
    pol1 = np.zeros((T, Na, Nx, Nb), dtype=np.int32)
    pwork = np.zeros((T, Na, Nx, Nb))
    c0 = np.zeros((T, Na, Nx, Nb))
    c1 = np.zeros((T, Na, Nx, Nb))
    tax0 = np.zeros((T, Na, Nx, Nb))
    tax1 = np.zeros((T, Na, Nx, Nb))
    V = np.zeros((T + 1, Na, Nx, Nb))
    V[T] = V_T

    t0 = time.time()
    for t in range(T - 1, -1, -1):
        w_h = float(seq["w_h"][t])
        r = float(seq["r"][t])
        lam = float(seq["lam"][t])
        gamma_t = float(seq["gamma"][t])
        transfer = float(seq["transfer"][t])
        d_total = float(seq["div"][t])

        dbar = d_total / Ex if Ex > 0 else 0.0
        div_x = dbar * x_grid

        EV = expected_value_next(V[t + 1], Px, Pbeta)
        Vh0 = np.full((Na, Nx, Nb), BIG_NEG)
        Vh1 = np.full((Na, Nx, Nb), BIG_NEG)

        for ia, a in enumerate(a_grid):
            for ix, x in enumerate(x_grid):
                for ib, b in enumerate(beta_grid):
                    for hflag in (0, 1):
                        h = params.hbar if hflag == 1 else 0.0
                        labor_income = w_h * x * h
                        ra = r * a
                        Ttax = total_tax(labor_income, ra, params.tau_k, lam, params.gamma_ss, gamma_t)
                        res = labor_income + (1.0 + r) * a - Ttax + transfer + div_x[ix]
                        best_val = BIG_NEG
                        best_ip = 0
                        best_c = 0.0
                        best_tax = Ttax
                        for ip, ap in enumerate(a_grid):
                            c = res - ap
                            if c <= 1e-14:
                                continue
                            val = np.log(c) - params.B * h + b * EV[ip, ix, ib]
                            if val > best_val:
                                best_val = val
                                best_ip = ip
                                best_c = c
                                best_tax = Ttax

                        if hflag == 0:
                            Vh0[ia, ix, ib] = best_val
                            pol0[t, ia, ix, ib] = best_ip
                            c0[t, ia, ix, ib] = best_c
                            tax0[t, ia, ix, ib] = best_tax
                        else:
                            Vh1[ia, ix, ib] = best_val
                            pol1[t, ia, ix, ib] = best_ip
                            c1[t, ia, ix, ib] = best_c
                            tax1[t, ia, ix, ib] = best_tax

        Vt, pt = logsumexp2(Vh0, Vh1, params.varrho)
        V[t] = Vt
        pwork[t] = pt

        if progress_cb is not None and ((T - t) == 1 or (T - t) % params.progress_transition_every_t == 0 or t == 0):
            progress_cb(f"[transition/backward] solved t={t} ({T-t}/{T}), elapsed={time.time()-t0:.1f}s")

    mu = np.zeros((T + 1, Na, Nx, Nb))
    mu[0] = mu0.copy()

    A = np.zeros(T + 1)
    N = np.zeros(T)
    C = np.zeros(T)
    Taxes = np.zeros(T)

    t1 = time.time()
    for t in range(T):
        A[t] = np.sum(mu[t] * a_grid[:, None, None])
        N[t] = np.sum(mu[t] * (x_grid[None, :, None] * params.hbar) * pwork[t])
        C[t] = np.sum(mu[t] * ((1.0 - pwork[t]) * c0[t] + pwork[t] * c1[t]))
        Taxes[t] = np.sum(mu[t] * ((1.0 - pwork[t]) * tax0[t] + pwork[t] * tax1[t]))

        mu_next = np.zeros_like(mu[t])
        for ia in range(Na):
            for ix in range(Nx):
                for ib in range(Nb):
                    mass = mu[t, ia, ix, ib]
                    if mass == 0:
                        continue
                    pw = pwork[t, ia, ix, ib]
                    ip0 = pol0[t, ia, ix, ib]
                    ip1 = pol1[t, ia, ix, ib]
                    for jx in range(Nx):
                        for jb in range(Nb):
                            prob = Px[ix, jx] * Pbeta[ib, jb]
                            mu_next[ip0, jx, jb] += mass * (1.0 - pw) * prob
                            mu_next[ip1, jx, jb] += mass * pw * prob
        mu[t + 1] = mu_next

        if progress_cb is not None and (t == 0 or (t + 1) % params.progress_transition_every_t == 0 or t == T - 1):
            progress_cb(f"[transition/forward] solved t={t} ({t+1}/{T}), elapsed={time.time()-t1:.1f}s")

    A[T] = np.sum(mu[T] * a_grid[:, None, None])

    return dict(V=V, mu=mu, A=A, N=N, C=C, Taxes=Taxes, pwork=pwork, pol0=pol0, pol1=pol1)
