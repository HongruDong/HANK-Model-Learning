"""
Pure-Python (no numpy/scipy/matplotlib) fallback runner.
目标：在依赖不可安装时，仍可跑出：
1) 举债支持公共基建的供给侧冲击下的通胀IRF
2) 核心机制变量IRF
3) 家庭异质性处理效应（按MPC/资产分组）
4) 税负分布×MPC/LPE协方差分解（需求侧通缩传导）
输出：CSV + SVG图
"""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass


@dataclass
class Config:
    T: int = 40
    # 政策冲击
    rho_Ig: float = 0.88
    Ig0: float = 0.015
    delta_g: float = 0.02
    psi_g: float = 0.22  # 公共资本对边际成本（或供给缺口）的影响强度

    # NK块
    beta: float = 0.995
    kappa: float = 0.05
    phi_pi: float = 1.7

    # 债务融资与挤出
    debt_crowdout: float = 0.70
    debt_speed: float = 0.20

    # 税负分配与协方差通道（新增）
    tax_adjust: float = 0.35          # 债务到税负调整强度
    tax_targeting_mpc: float = 0.80   # 税负偏向高MPC（>0）
    tax_targeting_lpe: float = 0.30   # 税负偏向高lpe（>0）

    # 异质性组：name, mpc, assets, lpe, weight
    groups: tuple = (
        ("constrained", 0.85, 0.05, 0.90, 0.30),
        ("middle", 0.55, 0.60, 0.55, 0.50),
        ("wealthy", 0.20, 2.50, 0.25, 0.20),
    )


def normalize_weights(groups):
    s = sum(g["weight"] for g in groups)
    if s <= 1e-12:
        raise RuntimeError("Group weights sum to zero")
    for g in groups:
        g["weight"] /= s
    return groups


def parse_groups(cfg: Config):
    out = []
    for row in cfg.groups:
        if len(row) == 5:
            name, mpc, assets, lpe, weight = row
        elif len(row) == 3:
            # backward compatibility
            name, mpc, assets = row
            lpe, weight = 0.5, 1.0
        else:
            raise RuntimeError("group tuple must be (name,mpc,assets,lpe,weight) or legacy length 3")
        out.append(dict(name=name, mpc=float(mpc), assets=float(assets), lpe=float(lpe), weight=float(weight)))
    return normalize_weights(out)


def wmean(values, weights):
    return sum(v * w for v, w in zip(values, weights))


def wcov(x, y, w):
    mx = wmean(x, w)
    my = wmean(y, w)
    return sum(wi * (xi - mx) * (yi - my) for xi, yi, wi in zip(x, y, w))


def solve_linear_system(A, b):
    """朴素高斯消元，适合小规模。"""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        if abs(pv) < 1e-12:
            raise RuntimeError("Singular matrix in linear solve")
        for j in range(col, n + 1):
            M[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            fac = M[r][col]
            if fac == 0:
                continue
            for j in range(col, n + 1):
                M[r][j] -= fac * M[col][j]
    return [M[i][n] for i in range(n)]


def build_paths(cfg: Config):
    T = cfg.T
    Ig = [0.0] * T
    Kg = [0.0] * T
    debt = [0.0] * T

    Ig[0] = cfg.Ig0
    for t in range(1, T):
        Ig[t] = cfg.rho_Ig * Ig[t - 1]

    for t in range(1, T):
        Kg[t] = (1.0 - cfg.delta_g) * Kg[t - 1] + Ig[t - 1]

    for t in range(1, T):
        debt[t] = (1.0 + cfg.debt_speed) * debt[t - 1] + Ig[t - 1]

    # 需求冲击项 u_t: 基建需求 - 债务挤出
    u = [Ig[t] - cfg.debt_crowdout * debt[t] for t in range(T)]
    # 供给项 s_t: 公共资本降低边际成本 => 在NKPC中相当于 +psi_g*Kg
    s = [cfg.psi_g * Kg[t] for t in range(T)]
    return Ig, Kg, debt, u, s


def solve_nk_irf(cfg: Config, u, s):
    """线性系统：
    pi_t = beta*pi_{t+1} + kappa*(x_t - s_t)
    x_t  = x_{t+1} - (phi_pi*pi_t - pi_{t+1}) + u_t
    """
    T = cfg.T
    n = 2 * T
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n

    def ipi(t): return t
    def ix(t): return T + t

    for t in range(T):
        # NKPC
        r = ipi(t)
        A[r][ipi(t)] = 1.0
        if t < T - 1:
            A[r][ipi(t + 1)] = -cfg.beta
        A[r][ix(t)] = -cfg.kappa
        b[r] = -cfg.kappa * s[t]

        # IS + Taylor
        r2 = ix(t)
        A[r2][ix(t)] = 1.0
        if t < T - 1:
            A[r2][ix(t + 1)] = -1.0
            A[r2][ipi(t + 1)] = -1.0
        A[r2][ipi(t)] = cfg.phi_pi
        b[r2] = u[t]

    sol = solve_linear_system(A, b)
    pi = sol[:T]
    x = sol[T:]
    p = []
    cum = 0.0
    for v in pi:
        cum += v
        p.append(cum)
    i_rate = [cfg.phi_pi * pi_t for pi_t in pi]
    r_real = [i_rate[t] - (pi[t + 1] if t < T - 1 else 0.0) for t in range(T)]
    return pi, x, p, i_rate, r_real


def heterogeneity_effects(cfg: Config, groups, pi, x, r_real):
    """简化处理效应：
    c_hat_g,t = mpc_g * x_t - eta_a * assets_g * r_real_t - eta_pi * pi_t
    """
    eta_a = 0.20
    eta_pi = 0.08
    out = {}
    for g in groups:
        c = []
        for t in range(cfg.T):
            c_hat = g["mpc"] * x[t] - eta_a * g["assets"] * r_real[t] - eta_pi * pi[t]
            c.append(c_hat)
        out[g["name"]] = c
    return out


def covariance_channel_decomposition(cfg: Config, groups, debt):
    """税负分布×MPC/lpe协方差分解。

    ΔC_direct ≈ -( mean(mpc)*mean(ΔT) + Cov(mpc, ΔT) )
    ΔN_direct ≈ -( mean(lpe)*mean(ΔT) + Cov(lpe, ΔT) )
    """
    T = cfg.T
    out = {
        "dT_base": [],
        "mean_dT": [],
        "cov_mpc_dT": [],
        "cov_lpe_dT": [],
        "avg_mpc_term": [],
        "avg_lpe_term": [],
        "direct_dC": [],
        "direct_dN": [],
        "group_dT": {g["name"]: [] for g in groups},
    }

    mpcs = [g["mpc"] for g in groups]
    lpes = [g["lpe"] for g in groups]
    ws = [g["weight"] for g in groups]
    mpc_bar = wmean(mpcs, ws)
    lpe_bar = wmean(lpes, ws)

    for t in range(T):
        dT_base = cfg.tax_adjust * debt[t]

        raw = []
        for g in groups:
            adj = 1.0
            adj += cfg.tax_targeting_mpc * (g["mpc"] - mpc_bar) / max(1e-8, mpc_bar)
            adj += cfg.tax_targeting_lpe * (g["lpe"] - lpe_bar) / max(1e-8, lpe_bar)
            raw.append(max(0.0, dT_base * adj))

        mean_raw = wmean(raw, ws)
        scale = (dT_base / max(1e-8, mean_raw)) if dT_base > 0 else 0.0
        dTs = [v * scale for v in raw]

        mean_dT = wmean(dTs, ws)
        cov_mpc = wcov(mpcs, dTs, ws)
        cov_lpe = wcov(lpes, dTs, ws)

        avg_mpc_term = mpc_bar * mean_dT
        avg_lpe_term = lpe_bar * mean_dT

        dC = -(avg_mpc_term + cov_mpc)
        dN = -(avg_lpe_term + cov_lpe)

        out["dT_base"].append(dT_base)
        out["mean_dT"].append(mean_dT)
        out["cov_mpc_dT"].append(cov_mpc)
        out["cov_lpe_dT"].append(cov_lpe)
        out["avg_mpc_term"].append(avg_mpc_term)
        out["avg_lpe_term"].append(avg_lpe_term)
        out["direct_dC"].append(dC)
        out["direct_dN"].append(dN)

        for g, dtg in zip(groups, dTs):
            out["group_dT"][g["name"]].append(dtg)

    return out


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_svg_line(path, title, series_dict):
    """简易SVG折线图（无网格）。"""
    width, height = 900, 420
    pad = 50
    T = len(next(iter(series_dict.values())))
    all_vals = [v for s in series_dict.values() for v in s]
    vmin, vmax = min(all_vals), max(all_vals)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    def sx(t):
        return pad + (width - 2 * pad) * t / max(1, T - 1)

    def sy(v):
        return height - pad - (height - 2 * pad) * (v - vmin) / (vmax - vmin)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    lines = []
    lines.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18">{title}</text>')
    lines.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#444"/>')
    lines.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#444"/>')

    # 坐标轴数值刻度（无网格）
    x_ticks = [0, max(0, T//4), max(0, T//2), max(0, 3*T//4), T-1]
    x_ticks = sorted(set(x_ticks))
    for xt in x_ticks:
        x = sx(xt)
        lines.append(f'<line x1="{x:.1f}" y1="{height-pad}" x2="{x:.1f}" y2="{height-pad+6}" stroke="#444"/>')
        lines.append(f'<text x="{x:.1f}" y="{height-pad+22}" text-anchor="middle" font-size="11">{xt}</text>')

    y_ticks = [vmin, (vmin+vmax)/2.0, vmax]
    for yv in y_ticks:
        y = sy(yv)
        lines.append(f'<line x1="{pad-6}" y1="{y:.1f}" x2="{pad}" y2="{y:.1f}" stroke="#444"/>')
        lines.append(f'<text x="{pad-10}" y="{y+4:.1f}" text-anchor="end" font-size="11">{yv:.3f}</text>')

    lines.append(f'<text x="{width/2:.1f}" y="{height-8}" text-anchor="middle" font-size="12">t</text>')

    for i, (name, s) in enumerate(series_dict.items()):
        pts = " ".join([f"{sx(t):.1f},{sy(s[t]):.1f}" for t in range(T)])
        col = colors[i % len(colors)]
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{pts}"/>')
        lines.append(f'<text x="{width-280}" y="{40 + 18*i}" fill="{col}" font-size="13">{name}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
{''.join(lines)}
</svg>'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def main():
    cfg = Config()
    groups = parse_groups(cfg)
    outdir = "output_fallback_no_deps"
    os.makedirs(outdir, exist_ok=True)

    Ig, Kg, debt, u, s = build_paths(cfg)
    pi, x, p, i_rate, r_real = solve_nk_irf(cfg, u, s)
    het = heterogeneity_effects(cfg, groups, pi, x, r_real)
    decomp = covariance_channel_decomposition(cfg, groups, debt)

    rows = []
    for t in range(cfg.T):
        rows.append([t, Ig[t], Kg[t], debt[t], u[t], s[t], pi[t], x[t], p[t], i_rate[t], r_real[t]])
    write_csv(
        os.path.join(outdir, "macro_irf.csv"),
        ["t", "Ig", "Kg", "debt", "u", "supply_term", "pi", "x", "price_level", "i", "r_real"],
        rows,
    )

    hrows = []
    for t in range(cfg.T):
        row = [t]
        for g in groups:
            row.append(het[g["name"]][t])
        hrows.append(row)
    write_csv(
        os.path.join(outdir, "heterogeneous_treatment_effects.csv"),
        ["t"] + [g["name"] for g in groups],
        hrows,
    )

    drows = []
    for t in range(cfg.T):
        drows.append([
            t,
            decomp["dT_base"][t],
            decomp["mean_dT"][t],
            decomp["avg_mpc_term"][t],
            decomp["cov_mpc_dT"][t],
            decomp["direct_dC"][t],
            decomp["avg_lpe_term"][t],
            decomp["cov_lpe_dT"][t],
            decomp["direct_dN"][t],
        ])
    write_csv(
        os.path.join(outdir, "fiscal_covariance_decomposition.csv"),
        ["t", "dT_base", "mean_dT", "avg_mpc_term", "cov_mpc_dT", "direct_dC", "avg_lpe_term", "cov_lpe_dT", "direct_dN"],
        drows,
    )

    gt_rows = []
    for t in range(cfg.T):
        row = [t]
        for g in groups:
            row.append(decomp["group_dT"][g["name"]][t])
        gt_rows.append(row)
    write_csv(
        os.path.join(outdir, "group_tax_burden_irf.csv"),
        ["t"] + [g["name"] for g in groups],
        gt_rows,
    )

    write_svg_line(os.path.join(outdir, "irf_inflation.svg"), "Inflation IRF (debt-financed public infrastructure)", {"pi": pi})
    write_svg_line(
        os.path.join(outdir, "irf_core_mechanisms.svg"),
        "Core mechanism IRFs",
        {"public capital Kg": Kg, "debt": debt, "demand u": u, "supply term": s, "output gap x": x, "real rate": r_real},
    )
    write_svg_line(
        os.path.join(outdir, "irf_heterogeneity.svg"),
        "Heterogeneous treatment effects (consumption response)",
        {g["name"]: het[g["name"]] for g in groups},
    )
    write_svg_line(
        os.path.join(outdir, "irf_covariance_channel.svg"),
        "Demand-side deflation channel: MPC/LPE covariance decomposition",
        {
            "-avg_mpc*mean_dT": [-v for v in decomp["avg_mpc_term"]],
            "-cov(mpc,dT)": [-v for v in decomp["cov_mpc_dT"]],
            "direct_dC": decomp["direct_dC"],
            "direct_dN": decomp["direct_dN"],
        },
    )
    write_svg_line(
        os.path.join(outdir, "irf_group_tax_burden.svg"),
        "Tax-burden IRFs by group",
        {g["name"]: decomp["group_dT"][g["name"]] for g in groups},
    )

    # summary
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Run completed with pure-Python fallback model.\n")
        f.write(f"Peak inflation response: {max(pi):.6f}\n")
        f.write(f"Trough inflation response: {min(pi):.6f}\n")
        f.write(f"Final price-level response: {p[-1]:.6f}\n")
        f.write(f"Min direct demand crowd-out (dC): {min(decomp['direct_dC']):.6f}\n")
        f.write(f"Peak cov(mpc,dT): {max(decomp['cov_mpc_dT']):.6f}\n")

    print("DONE", outdir)


if __name__ == "__main__":
    main()
