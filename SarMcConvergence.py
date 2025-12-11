import numpy as np
import pandas as pd
from scipy.stats import gamma
from matplotlib import pyplot as plt

# --- 1. 配置和输入参数 ---

# 选定的代表性 Gamma 参数 (来自 5.png 的 MLE 结果)
L_SHAPE = 5.4934
LAMBDA_RATE = 0.1067

# 模拟配置
K_REPETITIONS = 100  # 每个 N 值重复 K 次 (用于计算 MSE)
N_SIMULATIONS_LIST = np.logspace(3, 7, num=10, dtype=int)  # N = 10^3 to 10^7, 10 steps

# 待估计的统计量 (概率 P(I > t))
THRESHOLD_T = 100

# 输出配置
OUTPUT_CSV = 'mc_convergence_data.csv'
OUTPUT_PLOT = 'mc_mse_convergence.png'


# --- 2. 核心 MC 估计函数 ---

def run_single_mc_trial(N, L, lambda_rate, t_threshold):
    """
    执行单次 MC 估计，并返回其估计值和理论误差。
    """
    # Scipy 参数
    a, loc, scale = L, 0, 1.0 / lambda_rate

    # 1. 生成 N 个 Gamma 随机数
    mc_samples = gamma.rvs(a=a, loc=loc, scale=scale, size=N)

    # 2. MC 估计概率 P_MC(I > t)
    p_mc = np.sum(mc_samples > t_threshold) / N

    # 3. 理论真值 (解析解)
    p_theory = 1.0 - gamma.cdf(t_threshold, a=a, loc=loc, scale=scale)

    return p_mc, p_theory


# --- 3. 主收敛分析函数 ---

def main_convergence_analysis():
    results = []

    # 获取理论真值 (与 N 无关)
    _, P_THEORY_TRUE = run_single_mc_trial(1000, L_SHAPE, LAMBDA_RATE, THRESHOLD_T)
    print(f"Theoretical True Probability P(I > {THRESHOLD_T}): {P_THEORY_TRUE:.8f}")

    print(f"\n--- Starting Convergence Analysis (K={K_REPETITIONS}) ---")

    for N in N_SIMULATIONS_LIST:

        # 记录 K 次试验的误差平方和
        squared_errors = []

        for k in range(K_REPETITIONS):
            p_mc, _ = run_single_mc_trial(N, L_SHAPE, LAMBDA_RATE, THRESHOLD_T)

            # 计算误差平方 (用于计算 MSE)
            error = p_mc - P_THEORY_TRUE
            squared_errors.append(error ** 2)

        # 计算均方误差 (MSE)
        mse = np.mean(squared_errors)

        # 计算标准偏差 (用于绘制误差棒)
        std_dev = np.std(squared_errors) / np.sqrt(K_REPETITIONS)

        print(f"N={N:<10} | MSE={mse:.4e}")

        results.append({
            'N_Samples': N,
            'K_Repetitions': K_REPETITIONS,
            'MSE': mse,
            'MSE_StdDev': std_dev  # 用于误差棒
        })

    # 4. 数据保存
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nConvergence data saved to {OUTPUT_CSV}")

    # 5. 可视化 MSE 收敛曲线
    plot_convergence_curve(df, P_THEORY_TRUE)


# --- 4. 可视化函数 ---

def plot_convergence_curve(df, p_theory_true):
    """
    绘制 MSE 随 N 变化的 Log-Log 图。
    """

    plt.figure(figsize=(10, 6))

    # 绘制 MSE 随 N 的变化
    plt.errorbar(df['N_Samples'], df['MSE'], yerr=df['MSE_StdDev'],
                 fmt='o-', capsize=3, label='MC Estimation MSE')

    # 绘制理论收敛线 O(1/N)

    # 归一化常数 (在 Log-Log 图上找到一条斜率为 -1 的参考线)
    P_THEORY = p_theory_true
    C = P_THEORY * (1 - P_THEORY)

    # 在第一个数据点上绘制理论线
    theory_line = C / df['N_Samples']

    plt.plot(df['N_Samples'], theory_line, 'r--',
             label=f'Theoretical Rate $\mathcal{{O}}(1/N)$')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Monte Carlo Convergence Rate: MSE vs. Sample Size N (L={L_SHAPE:.2f})')
    plt.xlabel('Sample Size $N$ (Log Scale)')
    plt.ylabel('Mean Squared Error (MSE) - Log Scale')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(OUTPUT_PLOT)
    plt.close()
    print(f"Convergence plot saved to {OUTPUT_PLOT}")


if __name__ == '__main__':
    main_convergence_analysis()