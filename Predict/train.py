# train.py
# 训练脚本：多分位数预测 + 季节性分析

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from config import Config
from dataset import QuantileTrafficDataset, load_data, prepare_data_with_seasonality
from model import MultiQuantilePredictor, QuantileLoss, monotone_rearrangement
from seasonality import SeasonalityAnalyzer, create_seasonal_features
from copula import ResidualDependenceModeler


def reshape_batch_for_model(batch, device, config):
    """[新增] 重构批次以匹配 (B*N, S, F)"""

    # (B, N, S) -> (B*N, S, 1)
    seq_x = batch['seq_x'].to(device).reshape(-1, config.seq_len, 1)
    # (B, N, P) -> (B*N, P, 1)
    dec_input = batch['dec_input'].to(device).reshape(-1, config.pred_len, 1)
    # (B, N, P) -> (B*N, P)
    target = batch['target'].to(device).reshape(-1, config.pred_len)

    # (B, N, S+P) -> (B*N, S+P, 1)
    seasonal_prior = batch['seasonal_prior'].to(device).reshape(-1, config.seq_len + config.pred_len, 1)

    # (B, N, S+P, F_t) -> (B*N, S+P, F_t)
    # 假设F_t=6 (来自 seasonality.py)
    time_features = batch['time_features'].to(device).reshape(-1, config.seq_len + config.pred_len, 6)

    # (B, N)
    service_phases = batch['service_phases'].to(device)
    service_amplitudes = batch['service_amplitudes'].to(device)

    # (B*N)
    batch_size = service_phases.size(0)
    num_services = service_phases.size(1)
    service_indices = torch.arange(num_services).to(device).repeat(batch_size)

    return (
        seq_x, seasonal_prior, time_features,
        service_indices, service_phases, service_amplitudes,
        dec_input, target, batch_size, num_services
    )


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """[已重构] 训练一个epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        # [重构]
        (seq_x, seasonal_prior, time_features,
         service_indices, service_phases, service_amplitudes,
         dec_input, target, _, _) = reshape_batch_for_model(batch, device, config)

        optimizer.zero_grad()

        # 前向传播
        output = model(
            seq_x, seasonal_prior, time_features,
            service_indices, service_phases, service_amplitudes,
            dec_input
        )  # output: (B*N, P, Q)

        output = monotone_rearrangement(output, config.quantiles)

        loss = criterion(output, target)  # [重构] (B*N, P, Q) vs (B*N, P)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, config):
    """[已重构] 验证模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            # [重构]
            (seq_x, seasonal_prior, time_features,
             service_indices, service_phases, service_amplitudes,
             dec_input, target, _, _) = reshape_batch_for_model(batch, device, config)

            output = model(
                seq_x, seasonal_prior, time_features,
                service_indices, service_phases, service_amplitudes,
                dec_input
            )

            output = monotone_rearrangement(output, config.quantiles)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def plot_training_curve(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)

    if val_losses and any(val_losses):
        plt.plot(val_losses, label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Pinball Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  训练曲线已保存到: {save_path}")


def compute_residuals(model, data_loader, device, config):
    """
    [已重构] 计算残差用于Copula建模

    Returns:
        residuals_dict: {service: residual_array}
    """
    model.eval()

    # 收集每个服务的所有残差
    service_residuals_lists = {service: [] for service in config.service_cols}

    # 找到中位数索引
    median_idx = config.quantiles.index(0.5)

    with torch.no_grad():
        for batch in data_loader:
            # [重构]
            (seq_x, seasonal_prior, time_features,
             service_indices, service_phases, service_amplitudes,
             dec_input, target,
             batch_size, num_services) = reshape_batch_for_model(batch, device, config)

            # (B*N, P, Q)
            output = model(
                seq_x, seasonal_prior, time_features,
                service_indices, service_phases, service_amplitudes,
                dec_input
            )

            # 提取中位数预测 (B*N, P)
            median_pred = output[:, :, median_idx]

            # 计算残差: ε = y - ŷ_0.5
            # (B*N, P)
            residuals = target - median_pred

            # 将残差重塑回 (B, N, P)
            residuals_by_service = residuals.reshape(batch_size, num_services, config.pred_len)

            # 按服务分配
            for i, service in enumerate(config.service_cols):
                # (B, P) -> (B * P)
                res_np = residuals_by_service[:, i, :].cpu().numpy().flatten()
                service_residuals_lists[service].append(res_np)

    # 合并所有批次的残差
    residuals_dict = {}
    for service, res_list in service_residuals_lists.items():
        if res_list:
            residuals_dict[service] = np.concatenate(res_list)
        else:
            residuals_dict[service] = np.array([])

    return residuals_dict


def main():
    print("=" * 80)
    print("多分位数流量预测模型 - 训练模式")
    print("=" * 80)

    # [!] 关键: 确保 config 实例在全局可被辅助函数访问
    # 或者在调用时传递 (当前已在函数签名中包含 config)
    global config
    config = Config()

    # ==================== 1. 加载数据 ====================
    print("\n[1/7] 加载数据...")
    train_data_raw = load_data(config.train_file)
    print(f"  原始训练集: {len(train_data_raw)} 条记录")

    # 自动推断服务列
    print("  自动推断服务列...")
    all_cols = train_data_raw.columns.tolist()
    config.service_cols = [col for col in all_cols if col.endswith('_Arrivals')]
    config.num_services = len(config.service_cols)

    if config.num_services == 0:
        print("\n[!] 致命错误: 在数据文件中没有找到任何以 '_Arrivals' 结尾的服务列。")
        return

    print(f"  ✓ 发现 {config.num_services} 个服务列:")
    for col in config.service_cols:
        print(f"    - {col}")

    # 按时间戳聚合数据
    print("  正在按时间戳聚合所有地区的流量...")
    train_data_agg = train_data_raw.groupby('Timestamp')[config.service_cols].sum().reset_index()
    print(f"  聚合后训练集: {len(train_data_agg)} 条记录")
    print(f"  时间范围: {train_data_agg['Timestamp'].min()} 到 {train_data_agg['Timestamp'].max()}")

    # ==================== 2. 季节性分析 ====================
    print("\n[2/7] 执行季节性分析...")
    seasonality_analyzer = SeasonalityAnalyzer(
        period=config.period,
        smooth_weight=config.seasonality_smooth_weight
    )

    # 准备数据并执行季节性分析
    train_data, train_timestamps, seasonality_analyzer = prepare_data_with_seasonality(
        train_data_agg,
        config.service_cols,
        seasonality_analyzer=seasonality_analyzer,
        compute_seasonality=True
    )

    # 保存季节性参数
    seasonality_params = {
        'shared_template': seasonality_analyzer.shared_template,
        'service_params': seasonality_analyzer.service_seasonality,
        'period': seasonality_analyzer.period
    }
    joblib.dump(seasonality_params, config.seasonality_save_path)
    print(f"  ✓ 季节性参数已保存到: {config.seasonality_save_path}")

    # ==================== 3. 数据归一化 ====================
    print("\n[3/7] 数据归一化...")
    scaler = StandardScaler()
    train_data_scaled_values = scaler.fit_transform(train_data[config.service_cols])

    train_data_scaled = pd.DataFrame(train_data_scaled_values, columns=config.service_cols)
    train_data_scaled['Timestamp'] = train_data['Timestamp'].values

    print(f"  ✓ 数据归一化完成")

    # 保存Scaler
    joblib.dump(scaler, config.scaler_save_path)
    print(f"  ✓ Scaler已保存到: {config.scaler_save_path}")

    # ==================== 4. 创建时间特征 ====================
    print("\n[4/7] 创建时间特征...")
    time_features = create_seasonal_features(train_timestamps, period=config.period)
    print(f"  ✓ 时间特征形状: {time_features.shape}")

    # ==================== 5. 创建数据集 ====================
    print("\n[5/7] 创建数据集...")
    full_dataset = QuantileTrafficDataset(
        train_data_scaled,
        train_timestamps,
        config.seq_len,
        config.pred_len,
        config.service_cols,
        seasonality_analyzer=seasonality_analyzer,
        time_features=time_features
    )

    if len(full_dataset) == 0:
        print("\n[!] 致命错误: 训练数据集为空。")
        return

    # 划分数据集
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    if val_size == 0:
        print("  [!] 警告: 数据量太少，无法划分验证集。")
        train_subset = full_dataset
        val_subset = None
    else:
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size]
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = None
    if val_subset:
        val_loader = DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False
        )

    print(f"  训练样本数: {train_size}")
    print(f"  验证样本数: {val_size}")
    print(f"  批次大小: {config.batch_size}")

    # ==================== 6. 创建模型 ====================
    print("\n[6/7] 创建模型...")

    # [!] 确保 config.py 中的参数已被简化！
    # d_model=32, n_heads=2, e_layers=1, d_ff=64, service_emb_dim=8

    model = MultiQuantilePredictor(
        num_services=config.num_services,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        quantiles=config.quantiles,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        service_emb_dim=config.service_emb_dim
    ).to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数数量: {num_params:,}")
    print(f"  使用设备: {config.device}")
    print(f"  预测分位数: {config.quantiles}")

    # 损失函数和优化器
    criterion = QuantileLoss(config.quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # ==================== 7. 训练模型 ====================
    print(f"\n[7/7] 开始训练 ({config.epochs} epochs)...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        print(f"\nEpoch [{epoch + 1}/{config.epochs}]")

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device, config)
        train_losses.append(train_loss)

        # 验证
        val_loss = 0.0
        if val_loader:
            val_loss = validate(model, val_loader, criterion, config.device, config)
            val_losses.append(val_loss)
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            val_losses.append(0.0)
            print(f"  Train Loss: {train_loss:.6f} | (无验证集)")

        # 保存最佳模型
        current_loss_for_saving = val_loss if val_loader else train_loss
        if current_loss_for_saving < best_val_loss:
            best_val_loss = current_loss_for_saving

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'num_services': config.num_services,
                    'seq_len': config.seq_len,
                    'pred_len': config.pred_len,
                    'quantiles': config.quantiles,
                    'd_model': config.d_model,
                    'n_heads': config.n_heads,
                    'e_layers': config.e_layers,
                    'd_ff': config.d_ff,
                    'dropout': config.dropout,
                    'service_emb_dim': config.service_emb_dim,
                    'service_cols': config.service_cols,
                    'period': config.period
                }
            }, config.model_save_path)
            print(f"  ✓ 保存最佳模型 (Loss: {best_val_loss:.6f})")

    # ==================== 8. 残差分析和Copula建模 ====================
    print("\n[8/8] 执行残差分析和Copula建模...")

    # 计算残差
    print("  计算训练集残差...")
    # [!] 确保使用 val_loader 或 train_loader 进行残差计算
    # 这里使用 train_loader 来匹配原始逻辑
    residuals_dict = compute_residuals(model, train_loader, config.device, config)

    # 拟合t-Copula
    print("  拟合t-Copula...")
    copula_modeler = ResidualDependenceModeler(
        copula_df=config.copula_df,
        n_samples=config.copula_samples
    )
    copula_modeler.fit(residuals_dict)

    # 保存Copula参数
    copula_params = {
        'correlation_matrix': copula_modeler.correlation_matrix,
        'marginal_distributions': copula_modeler.marginal_distributions,
        'service_names': copula_modeler.service_names,
        'copula_df': copula_modeler.copula_df
    }
    joblib.dump(copula_params, config.copula_save_path)
    print(f"  ✓ Copula参数已保存到: {config.copula_save_path}")

    # ==================== 9. 保存结果 ====================
    print("\n[9/9] 保存训练结果...")

    # 绘制训练曲线
    plot_training_curve(
        train_losses,
        val_losses if val_loader else None,
        os.path.join(config.results_dir, 'training_curve.png')
    )

    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, config.epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_path = os.path.join(config.results_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"  训练历史已保存到: {history_path}")

    print("\n" + "=" * 80)
    print("训练完成!")
    print(f"最佳模型损失: {best_val_loss:.6f}")
    print(f"模型已保存到: {config.model_save_path}")
    print(f"季节性参数已保存到: {config.seasonality_save_path}")
    print(f"Copula参数已保存到: {config.copula_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()