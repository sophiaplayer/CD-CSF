# strategy.py
# — SNC 闭式上界 + 双时间尺度（代表分位基线 + In-Period 闭式更新）
# — 已纳入：线性正则 ε(式16)；慢层最小需求硬下界；Safe-Window 重配置预算
# — 关键点：用“带下界的单纯形投影”替代统一比例缩放，保证 θ ≥ θ_min 且 Σθ≤1
# — 预测读取严格对齐 test.py 输出的 quantile_predictions.csv

import os, math
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from data import (
    slice_1, slice_2, slice_3, slice_4, slice_5,
    TOTAL_COMPUTE_CAPACITY as F_n,
    TOTAL_BANDWIDTH as B_n,
    ALL_SLICES,
)

# ========== Slice 映射 ==========
SLICE = {
    "SMS_in": slice_1,
    "SMS_out": slice_2,
    "Call_in": slice_3,
    "Call_out": slice_4,
    "Internet": slice_5,
}

RESULTS_DIR = "results"
PRED_FILE_QUANT = os.path.join(RESULTS_DIR, "quantile_predictions.csv")
PRED_FILE_TIMELINE = os.path.join(RESULTS_DIR, "test_predictions_timeline.csv")
STRATEGY_OUTPUT_FILE = "resource_allocation_strategy.csv"

SERVICES = ["SMS_in", "SMS_out", "Call_in", "Call_out", "Internet"]

# ===================== 可调超参数 =====================

# MSI 控制周期
MSI_ALPHA_FOR_HORIZON = 0.90
MSI_ETA = 0.05
MSI_H_MAX = 24

# 全局预算分位数
BUDGET_ALPHA = 0.95

# Safe-window
SAFE_WINDOW_H = 24
SAFE_TARGET_VIOL = 0.05
SAFE_LAMBDA_GAMMA = 0.1
SAFE_ETA_GAMMA = 0.05
SAFE_P_MIN = 0.1
SAFE_P_MAX = 0.9

# SNC
SNC_EPSILON = 1e-5
SNC_EPSILON_PER_SERVICE = {
    # 可按 service_category 微调
}

# SNC surrogate 负载尺度
SNC_LAM0 = 1000.0

# 代表分位数 (和 test.py 的 quantiles 对应)
REP_QUANTILE = 0.90

# 计算与带宽分别使用不同的最小保障份额
MIN_GUAR_C = 0.001   # compute 侧 guaranteed 最小份额
MIN_GUAR_B = 0.0001  # bandwidth 侧 guaranteed 最小份额

REG_EPSILON = 1e-4
REG_EPSILON_PER_SERVICE = {
    # "media_analysis": 5e-4,
    # "batch_inference": 1e-3,
}

BETA_DEFAULT = 1.0
BETA_PER_SERVICE = {
    # 针对重载切片 batch_inference 提高 β 权重
    "batch_inference": 3.0,
    "media_analysis": 1.5,
}

EPSILON_LIN = 1e-6
D_MAX = 0.0007
D_MAX_PER_SERVICE = {
    # 允许 batch_inference 在快层中每步跨更大的份额
    "batch_inference": 0.002,
}

# Copula 校准的重配置预留因子（0 < RHO_RECONF < 1）
RHO_RECONF = 0.02

# 控制闭式解规模的缩放系数（相当于重新标定 β_t）
THETA_SCALE = 10.0

# dual 更新步长 & β_t 对 λ 的敏感度
MAX_DUAL_ITERS = 20
DUAL_STEP = 0.1
BETA_LAM_SCALE = 1e-4

# SNC 基线里的 θ_bar
THETA_BAR = 0.8

# soft-thresholding 收缩阈值
SOFT_THRESHOLD_TAU = 1e-4

# ===================== 工具函数 =====================

def _ensure_results_dir():
    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)

def _file_exists(path: str) -> bool:
    return os.path.isfile(path)

def _parse_timestamp(v):
    """
    转成毫秒级时间戳（int）
    """
    if isinstance(v, (int, float, np.integer, np.floating)):
        return int(v)
    dt = pd.to_datetime(v, errors="coerce", utc=False)
    if pd.isna(dt):
        raise RuntimeError(f"无法解析时间戳: {v}")
    return int(dt.value // 10**6)

# ===================== 时延分量 =====================

def tx_delay_at(theta_bw: float, data_mb: float) -> float:
    theta_bw = max(theta_bw, 1e-12)
    return data_mb / (theta_bw * B_n)

def exe_delay_at(theta_c: float, comp_cycles: float) -> float:
    theta_c = max(theta_c, 1e-12)
    return comp_cycles / (theta_c * F_n)

# ===================== SNC 队列上界 surrogate =====================

def snc_Wup(lam: float, theta_c: float, theta_bw: float, s) -> float:
    """
    数值稳定的 SNC 队列时延上界 surrogate：
      - 随 lam 增大而增大；
      - 随 theta_c 增大而减小；
    并用 delay_budget 设定尺度，避免把 D_compute_bar 压成负数。
    """
    lam = max(lam, 0.0)
    if lam <= 0:
        return 0.0

    base = 0.1 * float(s.delay_budget)  # 队列时延的基本尺度
    theta_c = max(theta_c, 1e-3)        # 防止过小
    load_factor = 1.0 + lam / SNC_LAM0  # 简单负载因子

    return base * load_factor / theta_c

# ===================== 联合需求读入 =====================

def load_joint_demand_quantile_series(alpha: float) -> Optional[np.ndarray]:
    """
    从 results/joint_demand_quantiles.csv 读取给定 alpha 的
    联合需求分位数时间序列 Q_{D_t}(alpha)。
    """
    jd_file = os.path.join(RESULTS_DIR, "joint_demand_quantiles.csv")
    if not _file_exists(jd_file):
        print(f"[WARN] 未找到 {jd_file}，联合需求轨迹不可用")
        return None

    try:
        df = pd.read_csv(jd_file)
    except Exception as e:
        print(f"[WARN] 读取 {jd_file} 失败: {e}")
        return None

    # 分位数列
    q_cols = [c for c in df.columns if c.startswith("q_")]

    # 兼容老格式: 列名本身是数值字符串 "0.9", "0.95"
    if not q_cols:
        for c in df.columns:
            if c == "Timestamp":
                continue
            try:
                float(c)
                q_cols.append(c)
            except Exception:
                continue

    if not q_cols:
        print(f"[WARN] {jd_file} 中找不到任何分位数字段")
        return None

    target_col = f"q_{alpha:.2f}"

    chosen_col = None
    if target_col in df.columns:
        chosen_col = target_col
    else:
        # 沿着所有分位数列找最接近 alpha 的
        best_col = None
        best_diff = None
        for c in q_cols:
            if c.startswith("q_"):
                suf = c.split("_", 1)[-1]
            else:
                suf = c
            try:
                qv = float(suf)
            except Exception:
                continue
            diff = abs(qv - alpha)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_col = c
        chosen_col = best_col

    if chosen_col is None:
        print(f"[WARN] 无法根据 alpha={alpha} 选择分位数列，联合需求轨迹不可用")
        return None

    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp")
    return df[chosen_col].astype(float).to_numpy()

def load_joint_demand_budget_and_series(
    default_capacity: float,
    alpha_budget: float = BUDGET_ALPHA,
) -> Tuple[float, Optional[np.ndarray]]:
    """
    使用联合需求分位数轨迹的分位数上界来校准全局预算 bar_C。
    """
    series = load_joint_demand_quantile_series(alpha_budget)
    if series is None or series.size == 0:
        print(f"[WARN] 无法从联合需求中校准预算，退回默认全局容量 F_n={default_capacity}")
        return default_capacity, None
    q_global = float(np.quantile(series, alpha_budget))
    bar_C = min(default_capacity, q_global)
    print(
        f"[INFO] 从联合需求轨迹的分位数上界校准全局预算: "
        f"q_global={q_global:.4f}, bar_C=min(F_n={default_capacity}, q_global)={bar_C:.4f}"
    )
    return bar_C, series

# ===================== MSI 控制周期 =====================

def compute_control_periods(
    Q_series: np.ndarray,
    eta: float,
    H: int,
    T: int,
) -> List[Tuple[int, int]]:
    """
    MSI 控制周期划分：保证窗口内相对波动 < eta。
    """
    periods: List[Tuple[int, int]] = []
    t = 0
    while t < T:
        Q_t = float(Q_series[t])
        if Q_t <= 0:
            k_star = 1
        else:
            k_max = min(H, T - t)
            window = Q_series[t: t + k_max]
            rel_dev = np.abs(window - Q_t) / max(abs(Q_t), 1e-9)
            mask = (rel_dev < eta)
            if not mask.any():
                k_star = 1
            else:
                k_star = int(np.argmax(~mask) + 1) if (~mask).any() else len(mask)
        start = t
        end = min(t + k_star - 1, T - 1)
        periods.append((start, end))
        t = end + 1
    print(f"[INFO] 自适应得到 {len(periods)} 个控制周期: {periods[:8]} ...")
    return periods

# ===================== In-Period 闭式控制 (P4 结构) =====================

def _get_reg_epsilon(service_category: str) -> float:
    return REG_EPSILON_PER_SERVICE.get(service_category, REG_EPSILON)

def _get_beta(service_category: str) -> float:
    return BETA_PER_SERVICE.get(service_category, BETA_DEFAULT)

def in_period_update_compute(
    lam_vec: np.ndarray,
    theta_prev_c: np.ndarray,
    theta_prev_b: np.ndarray,
    slices: List,
    budget: float = float('inf'),  # B_t: 重配置预算（绝对算力）
    cap_share: float = 1.0,        # 本时隙 compute 总份额上限 (<=1)
) -> np.ndarray:
    """
    Compute 侧 in-period 控制，对应论文中的 P4：
      min_θ Σ_s [ β_t C_s / (θ_s F_n) + λ_reg(s) θ_s ]
      s.t.
        - Σ_s θ_s <= cap_share
        - |θ_s - θ_prev_s| <= d_max
        - Σ_s (θ_s - θ_prev_s) F_n <= budget
        - θ_s >= MIN_GUAR_C
    使用对偶迭代 + 闭式 sqrt 更新。
    """
    S = len(slices)
    lam_vec = np.maximum(lam_vec, 0.0)
    theta_prev_c = np.clip(theta_prev_c, MIN_GUAR_C, 1.0)

    beta_base = np.array([_get_beta(s.service_category) for s in slices], dtype=float)
    reg_vec   = np.array([_get_reg_epsilon(s.service_category) for s in slices], dtype=float)
    C_vec     = np.array([s.compute_per_task for s in slices], dtype=float)
    dmax_vec  = np.array(
        [D_MAX_PER_SERVICE.get(s.service_category, D_MAX) for s in slices],
        dtype=float,
    )

    beta_t = beta_base * (1.0 + BETA_LAM_SCALE * lam_vec)

    lambda_cap = 0.0
    eta_reconf = 0.0

    theta_c = theta_prev_c.copy()
    cap_share = max(min(cap_share, 1.0), 0.0)

    for _ in range(MAX_DUAL_ITERS):
        for i, s in enumerate(slices):
            lam_i = float(max(0.0, lam_vec[i]))
            theta_prev_ci = float(theta_prev_c[i])
            theta_prev_bi = float(theta_prev_b[i])

            tx_prev  = tx_delay_at(theta_prev_bi, s.data_per_task_mb)
            exe_prev = exe_delay_at(theta_prev_ci, s.compute_per_task)
            w_prev   = snc_Wup(lam_i, theta_prev_ci, theta_prev_bi, s)
            if not np.isfinite(w_prev):
                w_prev = float("inf")
            total_bound_prev = tx_prev + exe_prev + w_prev

            if total_bound_prev > s.delay_budget:
                C_s    = C_vec[i]
                beta_i = beta_t[i]
                lam_reg = max(reg_vec[i], EPSILON_LIN)

                price = lam_reg + lambda_cap + eta_reconf * F_n
                price = max(price, EPSILON_LIN)

                theta_raw = THETA_SCALE * math.sqrt(
                    max(0.0, beta_i * C_s / (F_n * price))
                )

                low  = max(MIN_GUAR_C, theta_prev_ci - dmax_vec[i])
                high = min(1.0,       theta_prev_ci + dmax_vec[i])
                theta_c[i] = min(max(theta_raw, low), high)
            else:
                x = theta_prev_ci
                tau = SOFT_THRESHOLD_TAU
                if abs(x) <= tau:
                    theta_soft = 0.0
                else:
                    theta_soft = math.copysign(abs(x) - tau, x)

                low  = max(MIN_GUAR_C, theta_prev_ci - dmax_vec[i])
                high = min(1.0,       theta_prev_ci + dmax_vec[i])
                theta_c[i] = min(max(theta_soft, low), high)

        total_share = float(theta_c.sum())
        if total_share > cap_share + 1e-9:
            scale = cap_share / max(total_share, 1e-12)
            theta_c *= scale

        delta_comp = float((theta_c - theta_prev_c).sum() * F_n)
        if math.isfinite(budget):
            if budget <= 0.0 and delta_comp > 0.0:
                theta_c = theta_prev_c.copy()
                delta_comp = 0.0
            elif budget > 0.0 and delta_comp > budget:
                scale = budget / max(delta_comp, 1e-12)
                theta_c = theta_prev_c + (theta_c - theta_prev_c) * scale
                theta_c = np.clip(theta_c, MIN_GUAR_C, 1.0)
                delta_comp = float((theta_c - theta_prev_c).sum() * F_n)

        res_cap    = float(theta_c.sum() - cap_share)
        res_reconf = float(delta_comp - (budget if math.isfinite(budget) else delta_comp))

        lambda_cap = max(0.0, lambda_cap + DUAL_STEP * res_cap)
        if math.isfinite(budget):
            eta_reconf = max(0.0, eta_reconf + DUAL_STEP * res_reconf)

        if abs(res_cap) < 1e-4 and (not math.isfinite(budget) or abs(res_reconf) < 1e-2):
            break

    theta_c = np.clip(theta_c, MIN_GUAR_C, 1.0)
    return theta_c

def in_period_update_bandwidth(
    lam_vec: np.ndarray,
    theta_prev_b: np.ndarray,
    theta_prev_c: np.ndarray,
    slices: List,
    budget: float = float('inf'),
) -> np.ndarray:
    """
    带宽侧 in-period 控制：
      min_θ Σ_s [ β_t(s,t) * D_s / (θ_s B_n) + λ_reg(s) * θ_s ]
    约束:
      - Σ_s θ_s <= 1
      - |θ_s - θ_prev_s| <= d_max
      - Σ_s (θ_s - θ_prev_s) B_n <= budget
      - θ_s >= MIN_GUAR_B
    """

    S = len(slices)
    lam_vec = np.maximum(lam_vec, 0.0)
    theta_prev_b = np.clip(theta_prev_b, MIN_GUAR_B, 1.0)

    beta_base = np.array([_get_beta(s.service_category) for s in slices], dtype=float)
    reg_vec   = np.array([_get_reg_epsilon(s.service_category) for s in slices], dtype=float)
    D_vec     = np.array([s.data_per_task_mb for s in slices], dtype=float)
    dmax_vec  = np.array(
        [D_MAX_PER_SERVICE.get(s.service_category, D_MAX) for s in slices],
        dtype=float,
    )

    beta_t = beta_base * (1.0 + BETA_LAM_SCALE * lam_vec)

    lambda_cap = 0.0
    eta_reconf = 0.0
    theta_b = theta_prev_b.copy()

    for _ in range(MAX_DUAL_ITERS):
        for i, s in enumerate(slices):
            lam_i = float(max(0.0, lam_vec[i]))
            theta_prev_bi = float(theta_prev_b[i])
            theta_prev_ci = float(theta_prev_c[i])

            tx_prev  = tx_delay_at(theta_prev_bi, s.data_per_task_mb)
            exe_prev = exe_delay_at(theta_prev_ci, s.compute_per_task)
            w_prev   = snc_Wup(lam_i, theta_prev_ci, theta_prev_bi, s)
            if not np.isfinite(w_prev):
                w_prev = float("inf")
            total_bound_prev = tx_prev + exe_prev + w_prev

            if total_bound_prev > s.delay_budget:
                D_s = D_vec[i]
                beta_i = beta_t[i]
                lam_reg = max(reg_vec[i], EPSILON_LIN)

                price = lam_reg + lambda_cap + eta_reconf * B_n
                price = max(price, EPSILON_LIN)

                theta_raw = THETA_SCALE * math.sqrt(
                    max(0.0, beta_i * D_s / (B_n * price))
                )

                low  = max(MIN_GUAR_B, theta_prev_bi - dmax_vec[i])
                high = min(1.0,       theta_prev_bi + dmax_vec[i])
                theta_b[i] = min(max(theta_raw, low), high)
            else:
                x = theta_prev_bi
                tau = SOFT_THRESHOLD_TAU

                if abs(x) <= tau:
                    theta_soft = 0.0
                else:
                    theta_soft = math.copysign(abs(x) - tau, x)

                low  = max(MIN_GUAR_B, theta_prev_bi - dmax_vec[i])
                high = min(1.0,       theta_prev_bi + dmax_vec[i])
                theta_b[i] = min(max(theta_soft, low), high)

        total_share = float(theta_b.sum())
        if total_share > 1.0 + 1e-9:
            scale = 1.0 / max(total_share, 1e-12)
            theta_b *= scale

        delta_bw = float((theta_b - theta_prev_b).sum() * B_n)
        if math.isfinite(budget):
            if budget <= 0.0 and delta_bw > 0.0:
                theta_b = theta_prev_b.copy()
                delta_bw = 0.0
            elif budget > 0.0 and delta_bw > budget:
                scale = budget / max(delta_bw, 1e-12)
                theta_b = theta_prev_b + (theta_b - theta_prev_b) * scale
                theta_b = np.clip(theta_b, MIN_GUAR_B, 1.0)
                delta_bw = float((theta_b - theta_prev_b).sum() * B_n)

        res_cap    = float(theta_b.sum() - 1.0)
        res_reconf = float(delta_bw - (budget if math.isfinite(budget) else delta_bw))

        lambda_cap = max(0.0, lambda_cap + DUAL_STEP * res_cap)
        if math.isfinite(budget):
            eta_reconf = max(0.0, eta_reconf + DUAL_STEP * res_reconf)

        if abs(res_cap) < 1e-4 and (not math.isfinite(budget) or abs(res_reconf) < 1e-2):
            break

    theta_b = np.clip(theta_b, MIN_GUAR_B, 1.0)
    return theta_b

# ===================== SNC 基线：x_min / b_min =====================

def D_compute_bar(lam: float, s) -> float:
    return s.delay_budget - tx_delay_at(THETA_BAR, s.data_per_task_mb) - snc_Wup(lam, THETA_BAR, THETA_BAR, s)

def D_bw_bar(lam: float, s) -> float:
    return s.delay_budget - exe_delay_at(THETA_BAR, s.compute_per_task) - snc_Wup(lam, THETA_BAR, THETA_BAR, s)

def x_min_compute(lam: float, s) -> float:
    D = D_compute_bar(lam, s)
    return float("inf") if D <= 0 else s.compute_per_task / D

def b_min_bw(lam: float, s) -> float:
    D = D_bw_bar(lam, s)
    return float("inf") if D <= 0 else s.data_per_task_mb / D

# ===================== Safe-window: x_min & slack ξ_t & 预算 B_t =====================

def compute_xmin_and_slack_series(
    df_pred: pd.DataFrame,
    bar_C_series: np.ndarray,
    slices: List,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个时隙 t:
      - 计算各切片的 x_min(lam_{s,t})
      - 计算 slack ξ_t = bar_C_t - Σ_s x_min
    """
    T = len(df_pred)
    S = len(slices)
    x_min_ts = np.zeros((T, S), dtype=float)
    xi_series = np.zeros(T, dtype=float)

    for t in range(T):
        row = df_pred.iloc[t]
        total_demand = 0.0
        for i, s in enumerate(slices):
            lam = max(0.0, float(row[SERVICES[i]]))
            x_min = x_min_compute(lam, s)
            if not np.isfinite(x_min) or x_min <= 0:
                x_val = float(bar_C_series[t] + F_n)
            else:
                x_val = x_min
            x_min_ts[t, i] = x_val
            total_demand += x_val
        xi_series[t] = float(bar_C_series[t] - total_demand)
    return x_min_ts, xi_series

def identify_safe_windows_and_budgets(
    xi_series: np.ndarray,
    window: int,
    eps_target: float,
    lambda_gamma: float,
    eta_gamma: float,
    p_min: float,
    p_max: float,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    对应论文中 Safe-Window 识别与预算 B_t:

      - 利用滚动窗口 H 上的 slack 序列 {ξ_i} 估计违反率 \hat v_t
      - 自适应更新分位数水平 p_t 和阈值 γ_t
      - Safe set A = {t: ξ_t >= γ_t}
      - B_t = (1 - ρ) ξ_t (若 ξ_t < 0 则 B_t=0)
    """
    T = len(xi_series)
    safe_mask = np.zeros(T, dtype=bool)
    budgets = np.zeros(T, dtype=float)
    windows: List[Tuple[int, int]] = []

    if T == 0:
        return safe_mask, budgets, windows

    H = max(1, min(window, T))
    p = 0.5
    gamma = float(np.quantile(xi_series[:H], 0.5))

    for t in range(T):
        start = max(0, t - H + 1)
        window_xi = xi_series[start : t + 1]
        v_hat = float((window_xi < 0).mean())

        if v_hat > eps_target + eta_gamma:
            p = max(p_min, p - lambda_gamma)
        elif v_hat < eps_target - eta_gamma:
            p = min(p_max, p + lambda_gamma)

        gamma = float(np.quantile(window_xi, p))

        if xi_series[t] >= gamma:
            safe_mask[t] = True
            budgets[t] = max(0.0, (1.0 - rho) * xi_series[t])
        else:
            safe_mask[t] = False
            budgets[t] = 0.0

    start_idx = None
    for t in range(T):
        if safe_mask[t]:
            if start_idx is None:
                start_idx = t
        else:
            if start_idx is not None:
                windows.append((start_idx, t - 1))
                start_idx = None
    if start_idx is not None:
        windows.append((start_idx, T - 1))

    return safe_mask, budgets, windows

# ===================== 带下界的单纯形投影 =====================

def project_with_lower_bounds(
    theta_raw: np.ndarray,
    theta_min: np.ndarray,
    sum_limit: float = 1.0,
) -> np.ndarray:
    """
    求解:
      min ||θ - θ_raw||^2
      s.t. θ >= θ_min, Σ θ <= sum_limit
    """
    theta = np.maximum(theta_raw, theta_min)
    if theta.sum() <= sum_limit + 1e-12:
        return theta

    v = theta - theta_min
    if np.all(v <= 0):
        return theta_min.copy()

    idx = np.argsort(-v)
    v_sorted = v[idx]
    cumsum = np.cumsum(v_sorted)

    rho = -1
    theta_val = 0.0
    for j in range(len(v_sorted)):
        theta_j = (cumsum[j] - sum_limit + np.sum(theta_min)) / (j + 1)
        if v_sorted[j] > theta_j:
            rho = j
            theta_val = theta_j
    if rho == -1:
        return theta_min.copy()

    theta_proj = np.maximum(theta - theta_val, theta_min)
    return theta_proj

# ===================== 预测读取（对齐 test.py） =====================

def _parse_q_from_suffix(col: str) -> float:
    if "_q" not in col:
        return float("nan")
    suf = col.split("_q")[-1]
    try:
        return float(suf)
    except ValueError:
        return float("nan")

def load_predictions() -> Tuple[pd.DataFrame, List[str]]:
    """
    读取 quantile_predictions.csv 或 test_predictions_timeline.csv，
    并对每个服务选出代表分位数列。
    """
    if os.path.exists(PRED_FILE_QUANT):
        df = pd.read_csv(PRED_FILE_QUANT)
        if "Timestamp" not in df.columns:
            raise RuntimeError("quantile_predictions.csv 缺少 Timestamp 列")

        df["Timestamp"] = df["Timestamp"].apply(_parse_timestamp)

        out = pd.DataFrame()
        out["Timestamp"] = df["Timestamp"].values

        for s in SERVICES:
            cand_cols = [c for c in df.columns if c.startswith(f"{s}_q")]
            if not cand_cols:
                cand_cols = [c for c in df.columns if c.startswith(f"{s}_Arrivals_q")]
            chosen_col = None
            chosen_q = None

            if cand_cols:
                q_vals = []
                for c in cand_cols:
                    q = _parse_q_from_suffix(c)
                    q_vals.append(q)
                q_vals = np.array(q_vals, dtype=float)

                if np.isfinite(REP_QUANTILE) and len(q_vals) > 0:
                    idx = int(np.argmin(np.abs(q_vals - REP_QUANTILE)))
                    chosen_col = cand_cols[idx]
                    chosen_q = q_vals[idx]

                if chosen_col is None or not np.isfinite(chosen_q):
                    idx = int(np.argmax(q_vals))
                    chosen_col = cand_cols[idx]
                    chosen_q = q_vals[idx]
            else:
                if f"{s}_Arrivals_Predicted" in df.columns:
                    chosen_col = f"{s}_Arrivals_Predicted"
                    chosen_q = None
                elif s in df.columns:
                    chosen_col = s
                    chosen_q = None
                else:
                    raise RuntimeError(f"在 quantile_predictions.csv 中找不到与 {s} 对应的预测列")

            print(f"[load_predictions] {s} 使用列: {chosen_col} (q={chosen_q})")
            out[s] = df[chosen_col].astype(float).clip(lower=0.0)

        out["Total_Arrivals"] = out[SERVICES].sum(axis=1)
        return out, SERVICES

    if os.path.exists(PRED_FILE_TIMELINE):
        dft = pd.read_csv(PRED_FILE_TIMELINE)
        if "Timestamp" not in dft.columns:
            raise RuntimeError("test_predictions_timeline.csv 缺少 Timestamp 列")
        dft["Timestamp"] = dft["Timestamp"].apply(_parse_timestamp)
        for s in SERVICES:
            if s not in dft.columns:
                raise RuntimeError(
                    f"test_predictions_timeline.csv 中缺少 {s} 列")
        dft["Total_Arrivals"] = dft[SERVICES].sum(axis=1)
        return dft, SERVICES

    raise RuntimeError("找不到预测结果文件（quantile_predictions.csv 或 test_predictions_timeline.csv）")

# ===================== 主函数：生成策略 =====================

def generate_allocation_strategy():
    _ensure_results_dir()
    print("=== 生成策略（SNC + 双时间尺度 + MSI + Safe-Window + Dual P4） ===")

    # 1) 读取 per-service 预测 λ_{s,t}
    df_pred, _ = load_predictions()
    print(f"[INFO] 预测数据共 {len(df_pred)} 个时隙")

    slices = [SLICE[s] for s in SERVICES]
    S = len(slices)
    T = len(df_pred)

    # 2) 联合需求 -> 全局预算 bar_C（只用于 slack / Safe-Window，不再缩放 capacity share）
    bar_C, Q_budget = load_joint_demand_budget_and_series(F_n, alpha_budget=BUDGET_ALPHA)
    bar_C_series = np.full(T, bar_C, dtype=float)

    # 3) 计算 x_min_ts 和 slack ξ_t
    x_min_ts, xi_series = compute_xmin_and_slack_series(df_pred, bar_C_series, slices)

    # 4) Safe-Window
    safe_mask, budgets_compute, safe_windows = identify_safe_windows_and_budgets(
        xi_series=xi_series,
        window=SAFE_WINDOW_H,
        eps_target=SAFE_TARGET_VIOL,
        lambda_gamma=SAFE_LAMBDA_GAMMA,
        eta_gamma=SAFE_ETA_GAMMA,
        p_min=SAFE_P_MIN,
        p_max=SAFE_P_MAX,
        rho=RHO_RECONF,
    )
    print(f"[INFO] Safe-Window 数量: {len(safe_windows)}, windows={safe_windows[:8]} ...")

    # 5) MSI 控制周期（用 horizon 分位数 Q_{D_t}(MSI_ALPHA_FOR_HORIZON)）
    Q_horizon = load_joint_demand_quantile_series(MSI_ALPHA_FOR_HORIZON)
    if Q_horizon is not None and Q_horizon.size > 0:
        if Q_horizon.size < T:
            pad_val = Q_horizon[-1]
            Q_horizon = np.concatenate([Q_horizon, np.full(T - Q_horizon.size, pad_val)])
        else:
            Q_horizon = Q_horizon[:T]
        periods = compute_control_periods(
            Q_horizon,
            eta=MSI_ETA,
            H=MSI_H_MAX,
            T=T,
        )
    else:
        periods = [(0, T - 1)]
        print("[WARN] MSI horizon 缺失，退化为单一控制周期 [0, T-1]")

    print(f"[INFO] 控制周期: {periods[:8]} ... (共 {len(periods)} 个)")

    rows = []

    # 初始化 θ：均分后 clip 到各自的 MIN_GUAR_*
    theta_prev_c = np.clip(np.ones(S) / S, MIN_GUAR_C, 1.0)
    theta_prev_b = np.clip(np.ones(S) / S, MIN_GUAR_B, 1.0)

    # ---------- 6) 按控制周期遍历 ----------
    for (start_t, end_t) in periods:
        if start_t < 0 or end_t >= T:
            continue

        # 6.1 周期 envelope 负载 lam_eff
        lam_eff = np.zeros(S, dtype=float)
        for t in range(start_t, end_t + 1):
            row = df_pred.iloc[t]
            for i, s in enumerate(slices):
                lam_eff[i] = max(lam_eff[i], max(0.0, float(row[SERVICES[i]])))

        # 6.2 周期级 compute 总份额约束：归一化为 ∑_s θ_{s,k} ≤ 1
        bar_C_period = float(np.min(bar_C_series[start_t:end_t + 1]))
        cap_share_c_period = 1.0

        # 6.3 用 lam_eff 做 SNC：得到最小 compute/bw 需求，再转成 baseline 份额
        theta_base_c_raw = np.zeros(S, dtype=float)
        theta_base_b_raw = np.zeros(S, dtype=float)

        for i, s in enumerate(slices):
            lam_eff_i = max(0.0, float(lam_eff[i]))

            x_min = x_min_compute(lam_eff_i, s)
            b_min = b_min_bw(lam_eff_i, s)

            if np.isfinite(x_min) and x_min > 0:
                theta_base_c_raw[i] = min(1.0, max(MIN_GUAR_C, x_min / F_n))
            else:
                theta_base_c_raw[i] = MIN_GUAR_C

            if np.isfinite(b_min) and b_min > 0:
                theta_base_b_raw[i] = min(1.0, max(MIN_GUAR_B, b_min / B_n))
            else:
                theta_base_b_raw[i] = MIN_GUAR_B

        # 6.4 单纯形投影：baseline 只占掉 (1 - ρ) 的容量
        sum_limit_c_base = (1.0 - RHO_RECONF) * cap_share_c_period
        theta_base_c = project_with_lower_bounds(
            theta_base_c_raw,
            theta_min=np.full(S, MIN_GUAR_C, dtype=float),
            sum_limit=sum_limit_c_base,
        )

        sum_limit_b_base = (1.0 - RHO_RECONF) * 1.0
        theta_base_b = project_with_lower_bounds(
            theta_base_b_raw,
            theta_min=np.full(S, MIN_GUAR_B, dtype=float),
            sum_limit=sum_limit_b_base,
        )

        print(
            f"[INFO] 周期 [{start_t},{end_t}] baseline compute shares={theta_base_c}, "
            f"bandwidth shares={theta_base_b}"
        )

        # 6.5 周期内逐时隙快层控制
        for t in range(start_t, end_t + 1):
            r = df_pred.iloc[t]
            ts = int(r["Timestamp"])
            lam_t = np.array(
                [max(0.0, float(r[s])) for s in SERVICES],
                dtype=float,
            )

            # 本时隙 compute 总份额约束同样归一化为 1
            bar_C_t = float(bar_C_series[t])
            cap_share_c_t = 1.0

            # Safe-Window: 本时隙的重配置预算 B_t（不安全则 B_t=0）
            if 0 <= t < len(budgets_compute):
                budget_c = float(budgets_compute[t])
            else:
                budget_c = 0.0

            theta_c_raw = in_period_update_compute(
                lam_vec=lam_t,
                theta_prev_c=theta_prev_c,
                theta_prev_b=theta_prev_b,
                slices=slices,
                budget=budget_c,
                cap_share=cap_share_c_t,
            )

            theta_b_raw = in_period_update_bandwidth(
                lam_vec=lam_t,
                theta_prev_b=theta_prev_b,
                theta_prev_c=theta_prev_c,
                slices=slices,
                budget=float("inf"),  # 带宽暂不使用 Safe-Window 预算
            )

            # 以周期 baseline 作为“硬下界”做单纯形投影
            theta_c = project_with_lower_bounds(
                theta_raw=theta_c_raw,
                theta_min=theta_base_c,
                sum_limit=cap_share_c_t,
            )
            theta_b = project_with_lower_bounds(
                theta_raw=theta_b_raw,
                theta_min=theta_base_b,
                sum_limit=1.0,
            )
            theta_c = np.clip(theta_c, MIN_GUAR_C, 1.0)
            theta_b = np.clip(theta_b, MIN_GUAR_B, 1.0)

            # 更新状态
            theta_prev_c = theta_c.copy()
            theta_prev_b = theta_b.copy()

            # 输出行：Compute / Bandwidth 各有自己的 guaranteed 下界
            out = {"Timestamp": ts}
            for i, s in enumerate(slices):
                gid = s.service_id
                g_c = MIN_GUAR_C
                g_b = MIN_GUAR_B
                out[f"Slice_{gid}_Guaranteed_Compute"]   = g_c
                out[f"Slice_{gid}_Dynamic_Compute"]      = max(0.0, theta_c[i] - g_c)
                out[f"Slice_{gid}_Guaranteed_Bandwidth"] = g_b
                out[f"Slice_{gid}_Dynamic_Bandwidth"]    = max(0.0, theta_b[i] - g_b)
            rows.append(out)

    pd.DataFrame(rows).to_csv(STRATEGY_OUTPUT_FILE, index=False, float_format="%.8f")
    print(f"策略已保存：{STRATEGY_OUTPUT_FILE}\n=== 完成 ===")

if __name__ == "__main__":
    generate_allocation_strategy()
