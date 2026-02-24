# parameters.py
# 中文说明：模型参数集中管理。所有原文未明确、但对扩展“基建+债务→通缩”关键的要素，
# 一律在此文件中以 UNSPECIFIED 标注，并在注释中解释。

from dataclasses import dataclass
import numpy as np

UNSPECIFIED = "UNSPECIFIED"


def annual_to_quarterly_rate(r_annual: float) -> float:
    """年利率→季度利率（复利）。"""
    return (1.0 + r_annual) ** 0.25 - 1.0


@dataclass
class Parameters:
    # ===== 原文基准：偏好与异质性 =====
    hbar: float = 1.0 / 3.0
    B: float = 0.61
    varrho: float = 0.066  # Gumbel shock variance param in log-sum

    # Income risk: log x' = rho_x log x + eps, eps ~ N(0, sigma_x)
    rho_x: float = 0.939
    sigma_x: float = 0.287

    # Discount factor process (3-point Markov)
    beta_high: float = 0.9988
    delta_beta: float = 0.028
    pi_beta_stay: float = 0.995  # persistence on the beta grid

    # ===== 原文基准：技术与名义刚性 =====
    alpha: float = 0.36  # NOTE: in the paper, production is y=k^{1-alpha} n^{alpha}
    eps: float = 7.0
    Theta: float = 50.0

    eps_w: float = 7.0
    Theta_w: float = 50.0

    phi_k: float = 5.0

    # Depreciation: Table 1 says 0.0235, text says 0.035 -> UNSPECIFIED choice
    delta_table1: float = 0.0235
    delta_text: float = 0.035
    delta_use: str = "table1"  # "table1" or "text" (UNSPECIFIED selection)

    Pi_bar: float = 1.0

    # ===== 原文基准：财政与货币 =====
    tau_k: float = 0.35
    gamma_ss: float = 0.1
    lambda_ss: float = 0.68

    Transfers_ss: float = 0.11  # lump-sum transfers to households (steady ratio)
    G_ss: float = 0.13          # government purchases (steady ratio)
    D_ss: float = 1.33          # government debt (steady ratio)

    phi_Pi: float = 1.5

    # Government-spending shock persistence used in experiments
    rho_G: float = 0.90
    theta_deficit: float = 0.50  # share financed through deficits

    # Progressivity response: gamma_t - gamma = phi_progress * (G_t - G)
    phi_progress: float = 0.0  # 0 means constant progressivity

    # ===== 扩展：公共资本/基建投资 =====
    psi_g: float = 0.10     # public capital output elasticity (UNSPECIFIED; recommended starting point)
    delta_g: float = 0.02   # public capital depreciation (quarterly; UNSPECIFIED; adjust by data)
    phi_g: float = 0.0      # public investment adjustment cost (UNSPECIFIED; default 0)

    # Baseline public capital level (normalization)
    Kg_ss: float = 1.0

    # Public investment shock process: I_g shock AR(1)
    rho_Ig: float = 0.90
    Ig_shock_size: float = 0.01  # 1个标准差冲击（文献常用归一化）

    # Government purchases split: G^c + I^g
    Gc_ss: float = 0.13

    # ===== 可选扩展（未实现/需自行完善）=====
    debt_is_nominal: bool = False  # UNSPECIFIED: if True, model nominal debt B_t and real b_t=B_t/P_t
    debt_maturity: str = "one_period"  # UNSPECIFIED: "one_period" or "long_term"
    rho_B: str = UNSPECIFIED  # UNSPECIFIED if long-term debt is used
    zlb: bool = False         # UNSPECIFIED: add i_t >= 0 constraint

    # ===== 数值参数 =====
    T: int = 60
    Na: int = 120
    Nx: int = 7
    Nbeta: int = 3

    a_max: float = 60.0
    a_grid_curv: float = 3.0

    tol_vfi: float = 1e-6
    tol_vfi_rel: float = 1e-6   # 相对值函数收敛阈值
    tol_vfi_policy: int = 0      # 资产策略索引允许变动阈值（0=完全不变）
    minit_vfi_policy: int = 80   # 至少做这么多轮后才允许按策略稳定提前停止
    vfi_policy_stable_hits: int = 5  # 连续命中次数
    tol_vfi_policy_value: float = 1e-3  # 仅当diff也小于该阈值时，才允许按策略稳定提前停止
    tol_vfi_policy_value_rel: float = 1e-4  # 策略提前停止时的相对值函数阈值
    howard_start_hits: int = 3   # 策略稳定达到该次数后启用Howard策略评估
    howard_iter: int = 20        # 每次Howard策略评估迭代次数（0表示关闭）
    maxit_vfi: int = 2000
    tol_dist: float = 1e-12
    maxit_dist: int = 20000

    tol_outer: float = 1e-8
    maxit_outer: int = 60
    maxit_outer_ls: int = 120
    outer_stagnation_window: int = 12      # 连续多少次eval监测停滞
    outer_stagnation_tol: float = 5e-4     # 窗口内最优残差改善不足该值则判停滞
    solve_scale_Ew: float = 5.0            # 求解器内Ew缩放（仅数值求解，不改原残差定义）
    solve_scale_EG: float = 5.0            # 求解器内EG缩放（仅数值求解，不改原残差定义）
    outer_block_stagnation_tol: float = 1e-6  # Ew/EG块停滞判定阈值
    outer_max_eval_total: int = 80         # 外层总eval硬上限，避免长时间无休止运行
    outer_relaxed_accept_resid: float = 8e-1  # 阶段A先保证可收敛，再做精化

    # ===== 外层分块松弛迭代（兜底） =====
    block_relax_maxit: int = 20
    block_relax_omega_rk: float = 0.6
    block_relax_omega_wh: float = 0.6
    block_relax_omega_div: float = 0.6
    block_relax_omega_lam: float = 0.5
    block_relax_lam_min: float = 0.2
    block_relax_lam_max: float = 1.5


    # ===== 投影式固定点兜底 =====
    projected_fp_maxit: int = 150
    projected_fp_tol: float = 1e-2
    projected_fp_step_init: float = 0.5
    projected_fp_step_min: float = 1e-3
    projected_fp_omega_eq: float = 0.8
    projected_fp_omega_ew: float = 1.2
    projected_fp_omega_eg: float = 0.5
    projected_fp_omega_ed: float = 0.8

    # ===== 运行进度追踪（新增） =====
    progress_print: bool = True
    progress_vfi_every: int = 10
    progress_dist_every: int = 50
    progress_transition_every_t: int = 10
    progress_residual_eval_every: int = 1
    ss_root_round_digits: int = 6  # steady-state root缓存key的小数位

    # Steady-state interest-rate initial guess
    r_annual_guess: float = 0.035

    def delta(self) -> float:
        if self.delta_use == "table1":
            return self.delta_table1
        if self.delta_use == "text":
            return self.delta_text
        raise ValueError("delta_use must be 'table1' or 'text' (UNSPECIFIED choice).")

    def beta_grid(self) -> np.ndarray:
        bh = self.beta_high
        bm = bh - self.delta_beta
        bl = bh - 2.0 * self.delta_beta
        return np.array([bl, bm, bh], dtype=float)

    def wage_markup_ss(self) -> float:
        return self.eps_w / (self.eps_w - 1.0)

    def mc_ss(self) -> float:
        return (self.eps - 1.0) / self.eps

    def r_quarterly_guess(self) -> float:
        return annual_to_quarterly_rate(self.r_annual_guess)
