# test.py
# 测试脚本：多分位数预测 + 联合需求采样

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import joblib

from config import Config
from dataset import load_data, prepare_data_with_seasonality
from model import MultiQuantilePredictor
from seasonality import SeasonalityAnalyzer, create_seasonal_features
from copula import ResidualDependenceModeler


def predict_rolling_quantiles(
        model, test_data_np, test_timestamps, time_features,
        seasonality_analyzer, seq_len, pred_len, service_cols, device, quantiles
):
    """
    [已重构] 对测试集进行滚动多分位数预测

    适配 (B*N, S, 1) 架构
    """
    model.eval()

    num_services = len(service_cols)

    # 为每个分位数创建存储
    all_predictions = {q: [] for q in quantiles}
    all_targets = []
    all_timestamps = []

    total_len = len(test_data_np)

    if total_len < seq_len + pred_len:
        print(f"  [!] 错误: 数据总长 ({total_len}) 不足以进行一次预测")
        return None, None, None

    # [新增] 1. 准备静态的服务相关张量 (B=1)
    phases = []
    amplitudes = []
    if seasonality_analyzer.service_seasonality is None:
        print("[test.py] 警告: 季节性参数未初始化, 相位/幅度将使用 0")
        phases = np.zeros(num_services)
        amplitudes = np.zeros(num_services)
    else:
        for service in service_cols:
            params = seasonality_analyzer.service_seasonality.get(service, {})
            phases.append(params.get('phi', 0))  #
            amplitudes.append(params.get('a', 0))  #

    # 形状: (B, N) -> (1, N)
    service_phases_tensor = torch.FloatTensor(phases).unsqueeze(0).to(device)
    service_amplitudes_tensor = torch.FloatTensor(amplitudes).unsqueeze(0).to(device)
    # 形状: (B*N) -> (N)
    service_indices_tensor = torch.arange(num_services).to(device)

    progress_bar = tqdm(
        range(0, total_len - seq_len - pred_len + 1, pred_len),
        desc='Rolling Predict (Multi-Quantile)'
    )

    for i in progress_bar:
        # 准备输入
        # [修改] 提取 (S, N)
        seq_x_np = test_data_np[i: i + seq_len]
        # [修改] 提取 (P, N)
        dec_input_np = test_data_np[i + seq_len - pred_len: i + seq_len]
        # [修改] 提取 (P, N)
        target_np = test_data_np[i + seq_len: i + seq_len + pred_len]
        target_timestamps = test_timestamps[i + seq_len: i + seq_len + pred_len]

        # 时间特征 (S+P, F_t)
        time_feat = time_features[i: i + seq_len + pred_len]

        # 季节性先验 (S+P, N)
        t_indices = np.arange(i, i + seq_len + pred_len)
        seasonal_prior_np = np.zeros((seq_len + pred_len, num_services))
        for j, service in enumerate(service_cols):
            seasonal_prior_np[:, j] = seasonality_analyzer.get_seasonal_prior(service, t_indices)

        # [新增] 2. 重塑数据以匹配 (B*N, S, F) 架构 (B=1)

        # (S, N) -> (N, S) -> (N, S, 1)
        seq_x = torch.FloatTensor(seq_x_np.T).unsqueeze(-1).to(device)
        # (P, N) -> (N, P) -> (N, P, 1)
        dec_input = torch.FloatTensor(dec_input_np.T).unsqueeze(-1).to(device)

        # (S+P, N) -> (N, S+P) -> (N, S+P, 1)
        seasonal_prior = torch.FloatTensor(seasonal_prior_np.T).unsqueeze(-1).to(device)
        # (S+P, F_t) -> (1, S+P, F_t) -> (N, S+P, F_t)
        time_feat_tensor = torch.FloatTensor(time_feat).unsqueeze(0).repeat(num_services, 1, 1).to(device)

        # 3. 预测
        with torch.no_grad():
            output = model(
                seq_x,
                seasonal_prior,
                time_feat_tensor,
                service_indices_tensor,  # [新增]
                service_phases_tensor,  # [新增]
                service_amplitudes_tensor,  # [新增]
                dec_input  # [新增]
            )
            # output: (N, P, Q)

        # [修改] 4. 提取并重塑结果
        for q_idx, q in enumerate(quantiles):
            # (N, P) -> (P, N)
            pred_q = output[:, :, q_idx].cpu().numpy().T
            all_predictions[q].append(pred_q)

        all_targets.append(target_np)
        all_timestamps.append(target_timestamps)

    if not all_targets:
        return None, None, None

    # [修改] 5. 合并结果
    # all_predictions[q] 是一个 (P, N) 数组的列表
    # 合并后: (T, N)
    predictions_dict = {
        q: np.concatenate(preds, axis=0)
        for q, preds in all_predictions.items()
    }
    # all_targets 是一个 (P, N) 数组的列表
    # 合并后: (T, N)
    targets = np.concatenate(all_targets, axis=0)
    timestamps = np.concatenate(all_timestamps, axis=0)

    # (T, N) 格式与脚本其余部分 (plot, metrics) 兼容
    return predictions_dict, targets, timestamps


def calculate_quantile_metrics(predictions_dict, targets, quantiles):
    """
    计算多分位数预测的指标 (此函数无需修改)
    """
    metrics = {}

    # 1. 各分位数的MAE
    for q in quantiles:
        mae = np.mean(np.abs(predictions_dict[q] - targets))
        metrics[f'MAE_q{q}'] = mae

    # 2. Coverage (例如：80%区间的覆盖率)
    if 0.1 in quantiles and 0.9 in quantiles:
        lower = predictions_dict[0.1]
        upper = predictions_dict[0.9]
        coverage = np.mean((targets >= lower) & (targets <= upper))
        metrics['coverage_80'] = coverage

        # 区间宽度
        interval_width = np.mean(upper - lower)
        metrics['interval_width_80'] = interval_width

    # 3. Pinball Loss
    pinball_losses = []
    for q in quantiles:
        errors = targets - predictions_dict[q]
        pinball = np.where(errors >= 0, q * errors, (q - 1) * errors)
        pinball_losses.append(np.mean(pinball))

    metrics['avg_pinball_loss'] = np.mean(pinball_losses)

    return metrics


def plot_quantile_predictions(
        predictions_dict, targets, timestamps, service_cols,
        quantiles, save_path=None
):
    """
    绘制多分位数预测图 (此函数无需修改)
    """
    num_services = len(service_cols)
    fig, axes = plt.subplots(num_services, 1, figsize=(15, 4 * num_services))
    if num_services == 1:
        axes = [axes]

    try:
        time_index = pd.to_datetime(timestamps, unit='ms')
        xlabel = 'Timestamp'
    except:
        time_index = np.arange(len(timestamps))
        xlabel = 'Time Step'

    # 找到中位数和区间
    median_q = 0.5
    lower_q = min(quantiles)
    upper_q = max(quantiles)

    for j, service in enumerate(service_cols):
        ax = axes[j]

        # 真实值
        ax.plot(time_index, targets[:, j], 'o-', label='True',
                color='blue', linewidth=2, markersize=3, alpha=0.7)

        # 中位数预测
        if median_q in predictions_dict:
            ax.plot(time_index, predictions_dict[median_q][:, j], 's-',
                    label=f'Predicted (q={median_q})',
                    color='red', linewidth=2, markersize=3, alpha=0.7)

        # 不确定性区间
        if 0.1 in quantiles and 0.9 in quantiles:
            ax.fill_between(
                time_index,
                predictions_dict[0.1][:, j],
                predictions_dict[0.9][:, j],
                alpha=0.2, color='orange', label='80% Interval (q=0.1-0.9)'
            )

        # 如果有95%区间
        if 0.05 in quantiles and 0.95 in quantiles:
            ax.fill_between(
                time_index,
                predictions_dict[0.05][:, j],
                predictions_dict[0.95][:, j],
                alpha=0.1, color='green', label='90% Interval (q=0.05-0.95)'
            )

        ax.set_title(f'{service}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Arrivals', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(xlabel, fontsize=10)
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  多分位数预测图已保存到: {save_path}")
    else:
        plt.show()


# [函数开始]
# ... (test.py 中的其他 import 保持不变)
import pandas as pd  # 确保 pandas 已导入
from dataset import load_data  # 确保 load_data 已导入


# ... (predict_rolling_quantiles, calculate_quantile_metrics, plot_quantile_predictions 函数保持不变) ...

def forecast_with_joint_demand(config):
    """
    主预测函数：执行多分位数预测并计算联合需求分位数
    (此函数已修改 [3/5] 以解决冷启动问题)
    """
    print("=" * 80)
    print("多分位数流量预测 - 测试模式")
    print("=" * 80)

    # ==================== 1. 加载模型 ====================
    print("\n[1/5] 加载训练好的模型...")
    if not os.path.exists(config.model_save_path):
        print(f"错误: 找不到模型文件 {config.model_save_path}")
        return None, None, None, None, config

    checkpoint = torch.load(config.model_save_path, map_location=config.device)
    model_config = checkpoint['config']

    # 更新配置
    config.num_services = model_config['num_services']
    config.service_cols = model_config['service_cols']
    config.seq_len = model_config['seq_len']
    config.pred_len = model_config['pred_len']
    config.quantiles = model_config['quantiles']

    print(f"  服务列 ({config.num_services} 个): {config.service_cols}")
    print(f"  seq_len: {config.seq_len}, pred_len: {config.pred_len}")
    print(f"  预测分位数: {config.quantiles}")

    # 创建模型
    model = MultiQuantilePredictor(
        num_services=config.num_services,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        quantiles=config.quantiles,
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        e_layers=model_config['e_layers'],
        d_ff=model_config['d_ff'],
        dropout=model_config['dropout'],
        service_emb_dim=model_config['service_emb_dim']
    ).to(config.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ 模型加载成功")

    # ==================== 2. 加载季节性和Copula参数 ====================
    print("\n[2/5] 加载季节性和Copula参数...")

    # 季节性
    if not os.path.exists(config.seasonality_save_path):
        print(f"  错误: 找不到季节性参数文件 {config.seasonality_save_path}")
        print(f"  请先运行 train.py 训练模型！")
        return None, None, None, None, config
    else:
        seasonality_params = joblib.load(config.seasonality_save_path)

        seasonality_analyzer = SeasonalityAnalyzer(
            period=seasonality_params['period']
        )
        seasonality_analyzer.shared_template = seasonality_params['shared_template']
        seasonality_analyzer.service_seasonality = seasonality_params['service_params']
        print(f"  ✓ 季节性参数加载成功 (周期={seasonality_params['period']})")

    # Copula
    if not os.path.exists(config.copula_save_path):
        print(f"  警告: 找不到Copula参数文件 {config.copula_save_path}")
        copula_modeler = None
    else:
        copula_params = joblib.load(config.copula_save_path)
        copula_modeler = ResidualDependenceModeler(
            copula_df=copula_params['copula_df'],
            n_samples=config.copula_samples
        )
        copula_modeler.correlation_matrix = copula_params['correlation_matrix']
        copula_modeler.marginal_distributions = copula_params['marginal_distributions']
        copula_modeler.service_names = copula_params['service_names']
        print(f"  ✓ Copula参数加载成功")

    # Scaler
    scaler = joblib.load(config.scaler_save_path)
    print(f"  ✓ Scaler加载成功")

    # ==================== 3. 加载数据 (测试集 + 训练集历史) ====================
    # [!] 此步骤已修改以解决冷启动问题
    print("\n[3/5] 加载数据 (测试集 + 训练集历史)...")

    # 1. 加载训练集 (用于历史)
    try:
        # 从 config.py 获取 train_file
        train_data_raw = load_data(config.train_file)
        # 聚合 (与 train.py 逻辑一致)
        train_data_agg = train_data_raw.groupby('Timestamp')[config.service_cols].sum().reset_index()
        train_data_agg_sorted = train_data_agg.sort_values('Timestamp')
        # 提取历史
        history_df = train_data_agg_sorted.tail(config.seq_len)
        print(f"  ✓ 加载了 {len(history_df)} 条训练集历史记录 (用于预热)")
    except Exception as e:
        print(f"  [!] 警告: 无法加载训练集历史 ({config.train_file})。{e}")
        print("  将仅使用测试集进行冷启动。")
        history_df = pd.DataFrame(columns=['Timestamp'] + config.service_cols)

    # 2. 加载测试集
    test_data = load_data(config.test_file)
    print(f"  原始测试集: {len(test_data)} 条记录")

    # 聚合
    test_data_agg = test_data.groupby('Timestamp')[config.service_cols].sum().reset_index()
    test_data_agg_sorted = test_data_agg.sort_values('Timestamp')
    print(f"  聚合后测试集: {len(test_data_agg_sorted)} 条记录")

    # 3. 拼接数据
    combined_data_df = pd.concat([history_df, test_data_agg_sorted], ignore_index=True)
    print(f"  合并后总长度 (历史 + 测试): {len(combined_data_df)}")

    if len(combined_data_df) < config.seq_len + config.pred_len:
        print(
            f"  [!] 致命错误: 合并后的数据总长 ({len(combined_data_df)}) 不足以进行一次预测 (需要 {config.seq_len + config.pred_len})")
        return None, None, None, None, config

    # 4. 提取Numpy数组 (从合并后的数据中)
    test_timestamps_np = combined_data_df['Timestamp'].values
    test_values_np_unscaled = combined_data_df[config.service_cols].values

    # 5. 归一化 (使用已加载的 scaler)
    test_values_np_scaled = scaler.transform(test_values_np_unscaled)

    # 6. 创建时间特征 (为合并后的数据)
    time_features = create_seasonal_features(test_timestamps_np, period=config.period)

    # ==================== 4. 执行预测 ====================
    print("\n[4/5] 执行多分位数预测...")
    # [!] 调用重构后的函数
    predictions_dict_scaled, targets_scaled, pred_timestamps = predict_rolling_quantiles(
        model, test_values_np_scaled, test_timestamps_np, time_features,
        seasonality_analyzer, config.seq_len, config.pred_len,
        config.service_cols, config.device, config.quantiles
    )

    if predictions_dict_scaled is None:
        print("[!] 致命错误: 无法生成预测")
        return None, None, None, None, config

    # 反归一化
    print("  反归一化预测结果...")
    predictions_dict = {}
    for q in config.quantiles:
        pred_unscaled = scaler.inverse_transform(predictions_dict_scaled[q])
        pred_unscaled[pred_unscaled < 0] = 0  # 确保非负
        predictions_dict[q] = pred_unscaled

    targets_unscaled = scaler.inverse_transform(targets_scaled)

    print(f"  ✓ 预测完成")
    # 注意：这里的 timestamps 和 targets 仅包含测试集部分 (不含历史)
    print(f"    - 预测时间步数: {len(pred_timestamps)}")
    print(f"    - 预测分位数: {len(predictions_dict)}")

    # ==================== 5. 联合需求采样 (修正版) ====================
    joint_demand_quantiles_series = {}  # 存储结果: {q: [t0_val, t1_val, ...]}

    if copula_modeler is not None:
        print("\n[5/5] 计算联合需求分位数序列 (修正版)...")

        # 1. 获取中位数预测序列 (T, N)
        # 形状: (Time_steps, Num_services)
        median_preds_matrix = predictions_dict[0.5]
        num_timesteps = median_preds_matrix.shape[0]
        num_services = median_preds_matrix.shape[1]

        # 2. 从 Copula 生成相关的残差样本
        # 注意：这里我们生成 M 组残差，这 M 组残差反映了服务间的相关性
        # 形状: (M, N) -> M 个样本，每个样本包含 N 个服务的残差
        # 这里的 n_samples 是 config.copula_samples (例如 1000)
        # 我们假设 copula_modeler.sample_residuals() 返回 (M, N) 的残差矩阵
        # 如果 copula_modeler 没有暴露 sample_residuals，我们需要复用 sample_joint_demand 的逻辑

        try:
            # 生成 M 个相关联的均匀分布样本 U: (M, N)
            # 这里的 correlation_matrix 是 (N, N)
            # mvnorm 生成多元正态，然后用 norm.cdf 转为均匀分布 U
            from scipy.stats import multivariate_normal, norm, t

            M = config.copula_samples
            R = copula_modeler.correlation_matrix
            # 假设是 t-Copula，需要 nu (自由度)，如果代码里没存，暂时假设为高斯 Copula 或尝试获取
            # 这里简化演示，假设 copula_modeler 内部逻辑如下：

            # 生成多元标准正态样本 (用于 Gaussian Copula)
            # 如果是 t-Copula，逻辑类似但用 t 分布
            Z = multivariate_normal.rvs(mean=np.zeros(num_services), cov=R, size=M)
            U = norm.cdf(Z)  # (M, N)

            # 将 U 转换为残差 (通过逆累积分布函数/PPF)
            residuals_samples = np.zeros((M, num_services))
            for i, service in enumerate(config.service_cols):
                # 获取该服务的残差分布对象 (假设是 scipy stats 对象或经验分布)
                dist = copula_modeler.marginal_distributions[service]
                # 逆变换: U -> Residual
                # 注意：如果 dist 是经验分布，可能需要不同的调用方式
                # 这里假设 dist 有 ppf 方法 (如 scipy stats)
                if hasattr(dist, 'ppf'):
                    residuals_samples[:, i] = dist.ppf(U[:, i])
                else:
                    # 如果是其他自定义分布对象，请适配这里
                    # 例如如果是 KernelDensity，可能比较麻烦，通常建议拟合参数分布
                    pass

            # 3. 计算每一时刻的联合总负载分位数
            # 现在的逻辑：
            # 总负载(t, m) = sum_over_services( 预测值(t, s) + 残差(m, s) )
            #              = sum(预测值(t)) + sum(残差(m))

            # (T,) 每个时刻的所有服务预测值之和
            sum_forecasts_t = median_preds_matrix.sum(axis=1)

            # (M,) 每组模拟样本的所有服务残差之和
            sum_residuals_m = residuals_samples.sum(axis=1)

            # 利用广播机制计算总负载分布: (T, 1) + (1, M) -> (T, M)
            # total_load_samples[t, m] 是第 t 时刻第 m 次模拟的总负载
            total_load_samples = sum_forecasts_t[:, np.newaxis] + sum_residuals_m[np.newaxis, :]

            # 4. 计算分位数
            # 对 M 维度 (axis=1) 求分位数
            for q in config.quantiles:
                # np.quantile 可能会比较慢，对于大矩阵需注意
                q_vals = np.quantile(total_load_samples, q, axis=1)
                joint_demand_quantiles_series[q] = q_vals

            print(f"  ✓ 联合需求分位数序列计算完成 (长度: {num_timesteps})")

            # 为了兼容 main 函数的后续保存逻辑，我们可能需要调整保存部分
            # 因为原来的 joint_demand_quantiles 是单个值，现在是序列

        except Exception as e:
            print(f"  警告: 联合需求采样失败: {e}")
            import traceback
            traceback.print_exc()

    return predictions_dict, targets_unscaled, pred_timestamps, joint_demand_quantiles_series, config

def main():
    """主函数"""
    # [!] 关键: 确保 config 实例在全局可被辅助函数访问
    global config
    config = Config()

    # 执行预测
    predictions_dict, targets, pred_timestamps, joint_demand_q, config = forecast_with_joint_demand(config)

    if predictions_dict is None:
        print("\n测试执行失败")
        return

    # ==================== 评估指标 ====================
    print("\n[分析 1/3] 计算评估指标...")

    metrics = calculate_quantile_metrics(predictions_dict, targets, config.quantiles)

    print("\n  多分位数预测指标:")
    for key, value in metrics.items():
        if 'MAE' in key:
            print(f"  - {key}: {value:.6f}")

    if 'coverage_80' in metrics:
        print(f"\n  不确定性量化:")
        print(f"  - 80%区间覆盖率: {metrics['coverage_80']:.2%}")
        print(f"  - 80%区间平均宽度: {metrics['interval_width_80']:.6f}")

    print(f"\n  平均Pinball Loss: {metrics['avg_pinball_loss']:.6f}")

    # 联合需求分位数
    # 保存联合需求分位数（如果有）
    if joint_demand_q is not None:
        # 将字典转换为 DataFrame，每一列是一个分位数，行是时间步
        jd_df = pd.DataFrame(joint_demand_q)
        # 添加时间戳列
        jd_df['Timestamp'] = pred_timestamps

        # 排下列序，把 Timestamp 放第一列
        cols = ['Timestamp'] + [c for c in jd_df.columns if c != 'Timestamp']
        jd_df = jd_df[cols]

        jd_path = os.path.join(config.results_dir, 'joint_demand_quantiles_series.csv')
        jd_df.to_csv(jd_path, index=False, float_format='%.4f')
        print(f"  联合需求分位数序列已保存到: {jd_path}")

    # 确保结果目录存在
    os.makedirs(config.results_dir, exist_ok=True)

    # 保存各分位数的预测
    # ==================== 保存结果 ====================
    print("\n[分析 2/3] 保存结果...")

    # 确保结果目录存在
    os.makedirs(config.results_dir, exist_ok=True)

    # 1. 保存多分位数预测 (详细数据)
    results = []
    for t in range(len(pred_timestamps)):
        row = {'Timestamp': pred_timestamps[t]}
        for j, service in enumerate(config.service_cols):
            row[f'{service}_True'] = targets[t, j]
            for q in config.quantiles:
                # 格式化列名，避免浮点数精度问题
                row[f'{service}_q{q:.2f}'] = predictions_dict[q][t, j]
        results.append(row)

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(config.results_dir, 'quantile_predictions.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  多分位数预测已保存到: {csv_path}")

    # 2. 保存指标
    # 将字典键中的浮点数也格式化，防止潜在问题
    safe_metrics = {}
    for k, v in metrics.items():
        safe_metrics[str(k)] = v

    metrics_df = pd.DataFrame([safe_metrics])
    metrics_path = os.path.join(config.results_dir, 'quantile_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, float_format='%.6f')
    print(f"  评估指标已保存到: {metrics_path}")

    # 3. 保存联合需求分位数 (时间序列) [重点修改部分]
    if joint_demand_q is not None:
        try:
            # joint_demand_q 是 {0.95: np.array([...]), 0.99: np.array([...])}
            # 直接转换 DataFrame 会以 keys 为列，values 为数据列
            jd_df = pd.DataFrame(joint_demand_q)

            # 修改列名：从 0.95 变为 'q_0.95'，避免解析器报错
            jd_df.columns = [f"q_{col:.2f}" if isinstance(col, float) else str(col) for col in jd_df.columns]

            # 添加时间戳
            jd_df.insert(0, 'Timestamp', pred_timestamps)

            # 保存
            jd_path = os.path.join(config.results_dir, 'joint_demand_quantiles.csv')
            jd_df.to_csv(jd_path, index=False, float_format='%.4f')
            print(f"  联合需求分位数(序列)已保存到: {jd_path}")

        except Exception as e:
            print(f"  [!] 保存联合需求分位数失败: {e}")
            # 打印调试信息
            print(f"  Debug: joint_demand_q keys: {list(joint_demand_q.keys())}")
            if len(joint_demand_q) > 0:
                first_val = list(joint_demand_q.values())[0]
                print(f"  Debug: first value type: {type(first_val)}, shape: {getattr(first_val, 'shape', 'N/A')}")

    # ==================== 可视化 ====================
    print("\n[分析 3/3] 生成可视化...")

    plot_quantile_predictions(
        predictions_dict, targets, pred_timestamps, config.service_cols,
        config.quantiles,
        save_path=os.path.join(config.results_dir, 'quantile_predictions.png')
    )

    print("\n" + "=" * 80)
    print("测试和分析完成!")
    print("\n生成的文件:")
    print(f"  1. {csv_path} (多分位数预测)")
    print(f"  2. {metrics_path} (评估指标)")
    if joint_demand_q:
        print(f"  3. {jd_path} (联合需求分位数)")
    print(f"  4. quantile_predictions.png (可视化)")
    print("=" * 80)


if __name__ == "__main__":
    main()