# run_hank_public_capital.py
# 中文说明：主程序（含实时进度监控）

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from scipy.optimize import root, least_squares

from parameters import Parameters
from grids import make_asset_grid, stationary_dist_markov
from markov import tauchen, beta_markov_three_point
from household import solve_steady_state
from equilibrium import equilibrium_residual


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class OuterStagnationError(RuntimeError):
    pass


class RuntimeLogger:
    """同时输出到终端和日志文件，便于实时 tail 监控。"""

    def __init__(self, logfile: str, enabled: bool = True):
        self.enabled = enabled
        self.logfile = logfile
        if enabled:
            with open(self.logfile, "w", encoding="utf-8") as f:
                f.write("=== runtime progress log ===\n")

    def log(self, msg: str):
        if not self.enabled:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with open(self.logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def steady_state_solver(params: Parameters, grids, markov, logger: RuntimeLogger):
    a_grid = grids["a"]
    x_grid = grids["x"]
    beta_grid = grids["beta"]
    Px = markov["Px"]
    Pbeta = markov["Pbeta"]

    Mss = params.mc_ss()
    wage_markup = params.wage_markup_ss()
    delta = params.delta()

    z0 = np.array([1.0, params.r_quarterly_guess()])
    eval_count = {"n": 0}
    hh_cache = {}

    def _cache_key(w_h: float, r: float):
        d = params.ss_root_round_digits
        return (round(w_h, d), round(r, d))

    def _solve_hh_cached(w_h: float, r: float):
        key = _cache_key(w_h, r)
        if key not in hh_cache:
            hh_cache[key] = solve_steady_state(
                a_grid=a_grid,
                x_grid=x_grid,
                beta_grid=beta_grid,
                Px=Px,
                Pbeta=Pbeta,
                params=params,
                w_h=w_h,
                r=r,
                lam=params.lambda_ss,
                gamma_ss=params.gamma_ss,
                transfer=params.Transfers_ss,
                d_total=0.0,
                progress_cb=logger.log if params.progress_print else None,
            )
            if params.progress_print:
                logger.log(f"[steady/root] cache miss -> solved household at key={key}")
        else:
            if params.progress_print:
                logger.log(f"[steady/root] cache hit  -> reuse household at key={key}")
        return hh_cache[key]

    def F(z):
        eval_count["n"] += 1
        w_h, r = float(z[0]), float(z[1])
        if params.progress_print and eval_count["n"] % params.progress_residual_eval_every == 0:
            logger.log(f"[steady/root] eval={eval_count['n']}, w_h={w_h:.5f}, r={r:.5f}")
        if w_h <= 1e-6 or r <= -0.5:
            return np.array([1e3, 1e3])

        w = wage_markup * w_h
        u = r + delta
        M = (w / params.alpha) ** params.alpha * (u / (1.0 - params.alpha)) ** (1.0 - params.alpha)
        eq1 = M - Mss

        hh = _solve_hh_cached(w_h, r)
        A = hh["A"]
        N = hh["N"]
        K = A - params.D_ss
        if K <= 1e-10:
            eq2 = 1e3
        else:
            KN_house = K / max(1e-14, N)
            KN_firm = ((1.0 - params.alpha) / params.alpha) * (w / u)
            eq2 = KN_house - KN_firm

        if params.progress_print:
            logger.log(f"[steady/root] eval={eval_count['n']}, eq1={eq1:.3e}, eq2={eq2:.3e}")
        return np.array([eq1, eq2])

    logger.log("[steady] start root solve for (w_h, r)")
    sol = root(F, z0, method="hybr", tol=1e-10)
    logger.log(f"[steady] root done: success={sol.success}, message={sol.message}")

    w_h_ss, r_ss = float(sol.x[0]), float(sol.x[1])

    logger.log("[steady] recompute household steady state at converged (w_h, r)")
    hh_ss = solve_steady_state(
        a_grid=a_grid,
        x_grid=x_grid,
        beta_grid=beta_grid,
        Px=Px,
        Pbeta=Pbeta,
        params=params,
        w_h=w_h_ss,
        r=r_ss,
        lam=params.lambda_ss,
        gamma_ss=params.gamma_ss,
        transfer=params.Transfers_ss,
        d_total=0.0,
        progress_cb=logger.log if params.progress_print else None,
    )

    A_ss, N_ss = hh_ss["A"], hh_ss["N"]
    K_ss = A_ss - params.D_ss
    w_ss = params.wage_markup_ss() * w_h_ss
    qk_ss = 1.0
    Y_ss = (max(1e-14, K_ss) ** (1.0 - params.alpha)) * (max(1e-14, N_ss) ** params.alpha) * (params.Kg_ss ** params.psi_g)
    rk_ss = r_ss

    Phi = Y_ss - w_ss * N_ss - (rk_ss + params.delta() * qk_ss) * K_ss
    Phi_w = (w_ss - w_h_ss) * N_ss

    return dict(
        w_h_ss=w_h_ss,
        r_ss=r_ss,
        rk_ss=rk_ss,
        w_ss=w_ss,
        qk_ss=qk_ss,
        A_ss=A_ss,
        N_ss=N_ss,
        K_ss=K_ss,
        Y_ss=Y_ss,
        Phi=Phi,
        Phi_w=Phi_w,
        household_ss=hh_ss,
    )


def build_policy_paths(params: Parameters):
    """
    单一IRF政策路径：政府发债支持的1个标准差基建投资冲击。
    - Gc_t: 常数政府消费
    - Ig_t: t=0施加1个标准差冲击，之后按AR(1)衰减
    - transfer_t: 常数转移
    """
    T = params.T
    Gc = np.full(T, params.Gc_ss)

    Ig = np.zeros(T)
    Ig[0] = params.Ig_shock_size
    for t in range(1, T):
        Ig[t] = params.rho_Ig * Ig[t - 1]

    transfer = np.full(T, params.Transfers_ss)
    return dict(Gc=Gc, Ig=Ig, transfer=transfer)




def _residual_block_max(E: np.ndarray, T: int):
    return dict(
        Er=float(np.max(np.abs(E[0:T]))),
        Eq=float(np.max(np.abs(E[T:2*T]))),
        Ew=float(np.max(np.abs(E[2*T:3*T]))),
        EG=float(np.max(np.abs(E[3*T:4*T]))),
        Ed=float(np.max(np.abs(E[4*T:5*T]))),
    )


def solve_transition_equilibrium(params: Parameters, ss, grids, markov, policy, outdir: str, scenario: str, logger: RuntimeLogger):
    """
    外循环（已禁用Krylov）：
    1) 直接 least_squares
    2) 若停滞/预算耗尽或残差仍高，进入 block-relaxation 兜底
    """
    T = params.T

    rk0 = np.full(T, ss["rk_ss"])
    Pi0 = np.full(T, params.Pi_bar)
    wh0 = np.full(T, ss["w_h_ss"])
    lam0 = np.full(T, params.lambda_ss)
    d0 = np.zeros(T)
    X0 = np.concatenate([rk0, Pi0, wh0, lam0, d0], axis=0)

    progress = []
    eval_counter = {"n": 0}
    best = {"resid": np.inf, "x": X0.copy(), "E": None, "extras": None}
    resid_hist = []
    blk_hist = []

    def transition_progress(msg: str):
        if params.progress_print:
            logger.log(f"[{scenario}] {msg}")

    policy_monitored = dict(policy)
    policy_monitored["transition_progress_cb"] = transition_progress

    def _residual_only(x, log=True):
        eval_counter["n"] += 1
        t_start = time.time()
        if log and params.progress_print and eval_counter["n"] % params.progress_residual_eval_every == 0:
            logger.log(f"[{scenario}/outer] residual eval {eval_counter['n']} started")

        E, extras = equilibrium_residual(x, params, grids, markov, ss, policy_monitored)
        resid = float(np.max(np.abs(E)))
        blk = _residual_block_max(E, T)

        resid_hist.append(resid)
        blk_hist.append(blk)

        if resid < best["resid"]:
            best["resid"] = resid
            best["x"] = x.copy()
            best["E"] = E.copy()
            best["extras"] = extras

        if log and params.progress_print and eval_counter["n"] % params.progress_residual_eval_every == 0:
            logger.log(
                f"[{scenario}/outer] residual eval {eval_counter['n']} done, max|E|={resid:.3e}, "
                f"blocks={blk}, elapsed={time.time()-t_start:.1f}s"
            )
        return E, resid, extras, blk

    def _scale_for_solver(E: np.ndarray):
        Es = E.copy()
        Es[2 * T:3 * T] = Es[2 * T:3 * T] / max(1e-12, params.solve_scale_Ew)
        Es[3 * T:4 * T] = Es[3 * T:4 * T] / max(1e-12, params.solve_scale_EG)
        return Es

    def _check_stagnation():
        w = params.outer_stagnation_window
        if len(resid_hist) < w:
            return False
        recent = resid_hist[-w:]
        improve = max(recent) - min(recent)
        return improve < params.outer_stagnation_tol

    def _check_block_stagnation():
        w = params.outer_stagnation_window
        if len(blk_hist) < w:
            return False
        recent = blk_hist[-w:]
        ew = [b["Ew"] for b in recent]
        eg = [b["EG"] for b in recent]
        ew_improve = max(ew) - min(ew)
        eg_improve = max(eg) - min(eg)
        return (ew_improve < params.outer_block_stagnation_tol) and (eg_improve < params.outer_block_stagnation_tol)

    def _should_stop_by_budget():
        return eval_counter["n"] >= params.outer_max_eval_total

    def fun_ls(x):
        E, _, _, _ = _residual_only(x, log=True)
        scaled = _scale_for_solver(E)
        progress.append(float(np.max(np.abs(scaled))))
        if _should_stop_by_budget() or _check_stagnation() or _check_block_stagnation():
            raise OuterStagnationError(
                f"least_squares halted by guard: eval={eval_counter['n']}, best={best['resid']:.3e}"
            )
        return scaled

    def _block_relaxation(x_init: np.ndarray):
        logger.log(f"[{scenario}] start block-relaxation fallback")
        x = x_init.copy()
        for it in range(1, params.block_relax_maxit + 1):
            E, resid, extras, blk = _residual_only(x, log=True)
            if resid <= params.outer_relaxed_accept_resid:
                logger.log(f"[{scenario}] block-relax hit relaxed target at it={it}, resid={resid:.3e}")
                return x, E, extras, True, f"block_relax_converged_it={it}"

            rk = x[0:T].copy()
            Pi = x[T:2*T].copy()
            wh = x[2*T:3*T].copy()
            lam = x[3*T:4*T].copy()
            div = x[4*T:5*T].copy()

            Pi_w = extras["Pi_w"]
            w = extras["w"]
            N = extras["hh"]["N"]
            r = extras["r"]
            Y = extras["Y"]
            K = extras["K"]
            qk = extras["qk"]
            M = extras["M"]

            from equilibrium import wh_from_wage_pc, capital_foc_rk, profits_and_dividends

            wh_hat = wh_from_wage_pc(w, N, r, Pi_w, params.eps_w, params.Theta_w, params.Pi_bar)
            rk_hat = capital_foc_rk(M, Y, K[:T], qk, params.delta(), params.alpha)
            div_hat = profits_and_dividends(
                Y=Y, w=w, wh=wh, N=N, rk=rk, qk=qk, K=K, Pi=Pi, Pi_w=Pi_w, params=params,
                fixed_costs=(ss["Phi"], ss["Phi_w"])
            )

            wh = (1.0 - params.block_relax_omega_wh) * wh + params.block_relax_omega_wh * wh_hat
            rk = (1.0 - params.block_relax_omega_rk) * rk + params.block_relax_omega_rk * rk_hat
            div = (1.0 - params.block_relax_omega_div) * div + params.block_relax_omega_div * div_hat

            EG = E[3*T:4*T]
            lam = lam + params.block_relax_omega_lam * (EG / max(1e-8, (np.mean(np.abs(EG)) + 1e-8))) * 0.02
            lam = np.clip(lam, params.block_relax_lam_min, params.block_relax_lam_max)

            x = np.concatenate([rk, Pi, wh, lam, div], axis=0)
            logger.log(f"[{scenario}] block-relax it={it}, resid={resid:.3e}, blocks={blk}")

            if _should_stop_by_budget():
                logger.log(f"[{scenario}] block-relax stop by eval budget")
                break

        logger.log(f"[{scenario}] block-relax finished; best={best['resid']:.3e}")
        return best["x"].copy(), best["E"], best["extras"], False, "block_relax_best_returned"

    logger.log(f"[{scenario}] start outer least_squares solver (Krylov disabled)")
    logger.log(
        f"[{scenario}] outer_cfg: window={params.outer_stagnation_window}, tol={params.outer_stagnation_tol}, "
        f"eval_budget={params.outer_max_eval_total}, scale(Ew,EG)=({params.solve_scale_Ew},{params.solve_scale_EG}), "
        f"relaxed_accept={params.outer_relaxed_accept_resid:.3e}"
    )

    ls_success = False
    ls_message = ""
    x_ls = X0.copy()

    try:
        ls = least_squares(
            fun_ls,
            X0,
            method="trf",
            ftol=params.tol_outer,
            xtol=params.tol_outer,
            gtol=params.tol_outer,
            max_nfev=params.maxit_outer_ls,
            verbose=0,
        )
        ls_success = bool(ls.success)
        ls_message = f"least_squares_status={ls.status}"
        x_ls = ls.x
        logger.log(f"[{scenario}] least_squares done: success={ls.success}, status={ls.status}, best={best['resid']:.3e}")
    except OuterStagnationError as e:
        ls_success = False
        ls_message = f"least_squares_guard_stop: {e}"
        x_ls = best["x"].copy()
        logger.log(f"[{scenario}] least_squares interrupted: {e}")

    E_final, resid_final, extras_final, _ = _residual_only(x_ls, log=False)
    x_final = x_ls
    success_final = ls_success
    message_final = ls_message

    if resid_final > params.outer_relaxed_accept_resid and (not _should_stop_by_budget()):
        x_blk, E_blk, extras_blk, blk_ok, blk_msg = _block_relaxation(x_final)
        if E_blk is None:
            E_blk, _, extras_blk, _ = _residual_only(x_blk, log=False)
        resid_blk = float(np.max(np.abs(E_blk)))
        if resid_blk < resid_final:
            E_final, extras_final, x_final = E_blk, extras_blk, x_blk
            success_final = bool(blk_ok)
            message_final = blk_msg
            resid_final = resid_blk
            logger.log(f"[{scenario}] use block-relax solution resid={resid_final:.3e}")

    if progress:
        pd.DataFrame({
            "iter": np.arange(1, len(progress) + 1),
            "max_abs_scaled_resid": progress,
            "best_raw_resid": [best["resid"]] * len(progress),
        }).to_csv(os.path.join(outdir, f"solver_progress_{scenario}.csv"), index=False)

    def _projected_fixed_point(x_init, resid_init):
        logger.log(f"[{scenario}] start projected fixed-point fallback")
        x = x_init.copy()
        resid_curr = resid_init
        step = params.projected_fp_step_init
        for it in range(1, params.projected_fp_maxit + 1):
            E, resid, extras, _ = _residual_only(x, log=True)
            if resid < resid_curr:
                resid_curr = resid
            if resid <= params.projected_fp_tol:
                logger.log(f"[{scenario}] projected-FP converged at it={it}, resid={resid:.3e}")
                return x, E, extras, True, f"projected_fp_converged_it={it}"

            rk = x[0:T].copy()
            Pi = x[T:2*T].copy()
            wh = x[2*T:3*T].copy()
            lam = x[3*T:4*T].copy()
            div = x[4*T:5*T].copy()

            Eq = extras["Eq"]
            Ew = extras["Ew"]
            EG = extras["EG"]
            Ed = extras["Ed"]

            rk_prop = rk - step * params.projected_fp_omega_eq * Eq
            wh_prop = wh - step * params.projected_fp_omega_ew * Ew
            div_prop = div - step * params.projected_fp_omega_ed * Ed
            scale_eg = max(1e-8, float(np.mean(np.abs(EG))))
            lam_prop = lam + step * params.projected_fp_omega_eg * (EG / scale_eg) * 0.01
            lam_prop = np.clip(lam_prop, params.block_relax_lam_min, params.block_relax_lam_max)

            x_prop = np.concatenate([rk_prop, Pi, wh_prop, lam_prop, div_prop], axis=0)
            E_prop, resid_prop, extras_prop, _ = _residual_only(x_prop, log=False)

            if resid_prop < resid:
                x = x_prop
                resid_curr = resid_prop
                step = min(1.0, step * 1.1)
                logger.log(f"[{scenario}] projected-FP it={it}, accepted resid={resid_prop:.3e}, step={step:.3f}")
                E_final_loc, extras_final_loc = E_prop, extras_prop
            else:
                step *= 0.5
                logger.log(f"[{scenario}] projected-FP it={it}, rejected resid={resid_prop:.3e}, step={step:.3f}")
                if step < params.projected_fp_step_min:
                    break

            if _should_stop_by_budget():
                logger.log(f"[{scenario}] projected-FP stop by eval budget")
                break

        if best["E"] is not None:
            return best["x"].copy(), best["E"], best["extras"], False, "projected_fp_best_returned"
        return x, E_final_loc if 'E_final_loc' in locals() else E_final, extras_final_loc if 'extras_final_loc' in locals() else extras_final, False, "projected_fp_stopped"

    if resid_final > params.outer_relaxed_accept_resid and (not _should_stop_by_budget()):
        x_fp, E_fp, extras_fp, fp_ok, fp_msg = _projected_fixed_point(x_final, resid_final)
        resid_fp = float(np.max(np.abs(E_fp)))
        if resid_fp < resid_final:
            x_final, E_final, extras_final = x_fp, E_fp, extras_fp
            resid_final = resid_fp
            success_final = bool(fp_ok)
            message_final = fp_msg
            logger.log(f"[{scenario}] use projected-FP solution resid={resid_final:.3e}")

    if resid_final > params.outer_relaxed_accept_resid:
        logger.log(
            f"[{scenario}] WARNING: strict residual not achieved (resid={resid_final:.3e}), "
            f"return best available solution to avoid endless runtime"
        )

    sol_out = dict(x=x_final, success=bool(resid_final <= params.projected_fp_tol), message=message_final)
    return sol_out, E_final, extras_final

def save_plots_and_tables(params: Parameters, ss, extras, outdir: str, tag: str):
    ensure_dir(outdir)
    T = params.T
    t = np.arange(T)

    P_level = np.cumprod(extras["Pi"])

    df = pd.DataFrame(
        {
            "t": t,
            "Pi": extras["Pi"],
            "P_level": P_level,
            "r": extras["r"],
            "qk": extras["qk"],
            "Y": extras["Y"],
            "M": extras["M"],
            "w": extras["w"],
            "Pi_w": extras["Pi_w"],
            "goods_resid": extras["goods_resid"],
        }
    )
    df.to_csv(os.path.join(outdir, f"timeseries_{tag}.csv"), index=False)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(t, extras["Y"], linewidth=2)
    plt.axhline(ss["Y_ss"], linewidth=1)
    plt.title(f"产出路径（{tag}）")
    plt.xlabel("季度")
    plt.ylabel("Y")
    plt.grid(False)
    plt.xticks(np.arange(0, T, max(1, T // 10)))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"Y_{tag}.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(t, (extras["Pi"] - 1.0) * 100, linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.title(f"通胀路径（{tag}）")
    plt.xlabel("季度")
    plt.ylabel("pp")
    plt.grid(False)
    plt.xticks(np.arange(0, T, max(1, T // 10)))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"Pi_{tag}.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(t, (P_level - 1.0) * 100, linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.title(f"价格水平路径（{tag}）")
    plt.xlabel("季度")
    plt.ylabel("cum. pp")
    plt.grid(False)
    plt.xticks(np.arange(0, T, max(1, T // 10)))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"Plevel_{tag}.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(t, extras["goods_resid"], linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.title(f"商品市场残差诊断（{tag}）")
    plt.xlabel("季度")
    plt.ylabel("resid")
    plt.grid(False)
    plt.xticks(np.arange(0, T, max(1, T // 10)))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"goods_resid_{tag}.png"), dpi=150)
    plt.close(fig)


def main():
    params = Parameters()

    # 中文字体与负号显示设置（避免IRF图中文乱码）
    rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    outdir = "output"
    ensure_dir(outdir)
    logger = RuntimeLogger(os.path.join(outdir, "runtime_progress.log"), enabled=params.progress_print)

    logger.log("[main] start building grids + Markov")
    a_grid = make_asset_grid(params.Na, params.a_max, params.a_grid_curv)
    x_grid, Px = tauchen(params.rho_x, params.sigma_x, params.Nx, m=3.0)
    pi_x = stationary_dist_markov(Px)
    x_grid = x_grid / np.sum(pi_x * x_grid)

    Pbeta = beta_markov_three_point(params.pi_beta_stay)
    beta_grid = params.beta_grid()

    grids = dict(a=a_grid, x=x_grid, beta=beta_grid)
    markov = dict(Px=Px, Pbeta=Pbeta)

    logger.log("[main] start steady-state solver")
    ss = steady_state_solver(params, grids, markov, logger)
    logger.log(f"[main] steady state done: r_ss={ss['r_ss']:.6f}, Y_ss={ss['Y_ss']:.6f}")

    tag = "debt_financed_infrastructure_shock"
    logger.log(f"[main] run single IRF experiment: {tag}")
    policy = build_policy_paths(params)
    sol, E, extras = solve_transition_equilibrium(params, ss, grids, markov, policy, outdir=outdir, scenario=tag, logger=logger)
    logger.log(f"[main] IRF run done, success={sol['success']}, max|E|={np.max(np.abs(E)):.3e}")

    save_plots_and_tables(params, ss, extras, outdir=outdir, tag=tag)
    pd.DataFrame([
        dict(
            experiment=tag,
            success=bool(sol["success"]),
            max_abs_resid=float(np.max(np.abs(E))),
            Y0=float(extras["Y"][0]),
            Pi0=float(extras["Pi"][0]),
            goods_resid0=float(extras["goods_resid"][0]),
        )
    ]).to_csv(os.path.join(outdir, "irf_summary.csv"), index=False)

    irf_df = pd.DataFrame({
        "t": np.arange(params.T),
        "Ig": policy["Ig"],
        "Kg": extras["Kg"][:params.T],
        "Pi_irf_pp": (extras["Pi"] - 1.0) * 100,
        "Y_irf": extras["Y"] - ss["Y_ss"],
        "r_irf_pp": (extras["r"] - ss["r_ss"]) * 100,
        "goods_resid": extras["goods_resid"],
    })
    irf_df.to_csv(os.path.join(outdir, "irf_core_series.csv"), index=False)

    logger.log("[main] single IRF experiment completed")
    print(irf_df.head())


if __name__ == "__main__":
    main()
