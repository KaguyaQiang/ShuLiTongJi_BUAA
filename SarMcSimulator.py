import numpy as np
import pandas as pd
from scipy.stats import gamma
import os

# --- 1. 配置和输入/输出文件设置 ---
INPUT_CSV = 'sar_mle_results.csv'
OUTPUT_CSV = 'sar_mc_validation_results.csv'

# --- 2. 模拟参数定义 ---
N_SIMULATIONS = 10_000_000  # 设置 MC 模拟次数
THRESHOLD_T = 100  # 概率估计的阈值 t
PROBABILITY_P = 0.999  # 分位数估计的概率 p


# --- 3. MC 估计函数 ---

def estimate_probability_and_quantile(L, lambda_rate, N_sim, threshold_t, p_quantile):
    """
    对给定的 Gamma 参数执行 MC 模拟，并计算概率和分位数。
    """
    # 转换为 Scipy 所需的 (a, loc, scale) 格式
    a = L
    loc = 0
    scale = 1.0 / lambda_rate

    # 1. 生成 N 个服从 Gamma(L, lambda) 分布的随机数
    mc_samples = gamma.rvs(a=a, loc=loc, scale=scale, size=N_sim)

    # --- A. 概率估计 P(I > t) ---
    count_greater_t = np.sum(mc_samples > threshold_t)
    p_mc = count_greater_t / N_sim
    p_theory = 1.0 - gamma.cdf(threshold_t, a=a, loc=loc, scale=scale)

    # --- B. 分位数估计 Q(p) ---
    sorted_samples = np.sort(mc_samples)
    index = int(np.ceil(p_quantile * N_sim)) - 1
    q_mc = sorted_samples[index]
    q_theory = gamma.ppf(p_quantile, a=a, loc=loc, scale=scale)

    # 返回 MC 结果和理论真值
    return {
        'P_MC': p_mc,
        'P_Theory': p_theory,
        'Q_MC': q_mc,
        'Q_Theory': q_theory
    }


# --- 4. 主自动化函数 ---

def main_automation():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found. Please run SarRoiAnalysis.py first.")
        return

    # 读取 MLE 结果
    df_mle = pd.read_csv(INPUT_CSV)

    mc_results = []

    print(f"--- 阶段二：对 {len(df_mle)} 组参数进行 MC 模拟 (N={N_SIMULATIONS / 1e6:.0f}M) ---")

    for index, row in df_mle.iterrows():
        img_name = row['Image_Name']
        L_hat = row['L_hat (Shape)']
        lambda_hat = row['Lambda_hat (Rate)']

        print(f"Processing {img_name} (L={L_hat:.3f}, Lambda={lambda_hat:.4f})...")

        # 执行 MC 模拟和估计
        results = estimate_probability_and_quantile(
            L_hat, lambda_hat, N_SIMULATIONS, THRESHOLD_T, PROBABILITY_P
        )

        # 合并结果
        result_row = {
            'Image_Name': img_name,
            'L_hat': L_hat,
            'Lambda_hat': lambda_hat,
            'N_MC': N_SIMULATIONS,
            'T_Threshold': THRESHOLD_T,
            'P_Quantile': PROBABILITY_P,

            'P_MC': results['P_MC'],
            'P_Theory': results['P_Theory'],
            'P_Abs_Error': np.abs(results['P_MC'] - results['P_Theory']),

            'Q_MC': results['Q_MC'],
            'Q_Theory': results['Q_Theory'],
            'Q_Abs_Error': np.abs(results['Q_MC'] - results['Q_Theory']),
        }
        mc_results.append(result_row)

    # 保存最终结果
    df_mc = pd.DataFrame(mc_results)
    df_mc.to_csv(OUTPUT_CSV, index=False)

    print("\n--- MC 自动化分析完成 ---")
    print(f"详细 MC 验证结果已保存到 {OUTPUT_CSV}")


if __name__ == '__main__':
    main_automation()