# model.py
# 多分位数流量预测模型

import torch
import torch.nn as nn
import numpy as np
import math


class QuantileLoss(nn.Module):
    """
    Pinball Loss (分位数损失) - 向量化实现
    """

    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        # 形状: (1, 1, n_quantiles)
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32).view(1, 1, -1)

    def forward(self, predictions, targets):
        """
        [已重构]
        Args:
            predictions: (batch, pred_len, n_quantiles)
            targets: (batch, pred_len)
        Returns:
            loss: 平均Pinball损失 (一个标量)
        """
        if self.quantiles.device != predictions.device:
            self.quantiles = self.quantiles.to(predictions.device)

        # 1. 扩展 targets
        # (batch, pred_len) -> (batch, pred_len, 1)
        targets_expanded = targets.unsqueeze(2)

        # 2. 计算误差 (B, P, Q)
        errors = targets_expanded - predictions

        # 3. 计算分位数损失 (B, P, Q)
        loss = torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors
        )

        return loss.mean()


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ServiceEmbedding(nn.Module):
    """
    [已重构] ServiceEmbedding
    现在接收 (B, N) 的相位/幅度，并为 (B*N) 的批次准备 Embedding
    """

    def __init__(self, num_services, emb_dim):
        super(ServiceEmbedding, self).__init__()
        self.num_services = num_services
        self.emb_dim = emb_dim

        # 可学习的服务embedding (meta_s)
        # 形状: (N, E_dim)
        self.service_emb = nn.Parameter(torch.randn(num_services, emb_dim))

        # 用于处理相位和幅度的投影层
        self.phase_proj = nn.Linear(1, emb_dim // 4)
        self.amplitude_proj = nn.Linear(1, emb_dim // 4)

        base_dim = emb_dim
        phase_dim = emb_dim // 4
        amp_dim = emb_dim // 4
        self.final_proj = nn.Linear(base_dim + phase_dim + amp_dim, emb_dim)

    def forward(self, service_indices, phases, amplitudes):
        """
        Args:
            service_indices: (B*N) 索引 0...N-1
            phases: (B, N) 相位参数 φ_s
            amplitudes: (B, N) 幅度参数 a_s

        Returns:
            embeddings: (B*N, emb_dim)
        """
        # (B, N) -> (B*N, 1)
        phases_flat = phases.reshape(-1, 1)
        amplitudes_flat = amplitudes.reshape(-1, 1)

        # (N, E_dim) -> (B*N, E_dim)
        base_emb = self.service_emb[service_indices]

        # (B*N, 1) -> (B*N, E_dim/4)
        phase_emb = self.phase_proj(phases_flat)
        amp_emb = self.amplitude_proj(amplitudes_flat)

        # 拼接 (B*N, E_dim + E/4 + E/4)
        combined_emb = torch.cat([base_emb, phase_emb, amp_emb], dim=-1)

        # 投影回 (B*N, E_dim)
        return self.final_proj(combined_emb)


class MultiQuantilePredictor(nn.Module):
    """
    [已重构] 多分位数流量预测器

    现在模型一次只处理一个服务
    f_θ(x_t, e_s, Ŝ^prior_{s,t})
    """

    def __init__(
            self,
            num_services,  # 仅用于ServiceEmbedding
            seq_len,
            pred_len,
            quantiles,
            d_model=32,
            n_heads=2,
            e_layers=1,
            d_ff=64,
            dropout=0.1,
            service_emb_dim=8,
            seasonal_feature_dim=6
    ):
        super(MultiQuantilePredictor, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.d_model = d_model

        # 服务Embedding
        self.service_embedding = ServiceEmbedding(num_services, service_emb_dim)

        # [重构] 输入投影
        # 输入: [时序 (1) + 季节性先验 (1) + 时间特征 (F_t) + 服务Emb (E_dim)]
        input_dim = 1 + 1 + seasonal_feature_dim + service_emb_dim
        self.input_projection = nn.Linear(input_dim, d_model)

        # 解码器输入投影
        # 输入: [时序 (1) + 季节性先验 (1) + 时间特征 (F_t) + 服务Emb (E_dim)]
        self.decoder_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # [重构] 输出头
        # 输出: (B, P, Q)
        # 每个服务只预测自己的多分位数
        self.quantile_head = nn.Linear(d_model, self.n_quantiles)

    def forward(self,
                seq_x_flat,
                seasonal_prior_flat,
                time_features_flat,
                service_indices,
                service_phases,
                service_amplitudes,
                dec_input_flat
                ):
        """
        [已重构]
        Args:
            seq_x_flat: (B*N, S, 1) 输入序列
            seasonal_prior_flat: (B*N, S+P, 1) 季节性先验
            time_features_flat: (B*N, S+P, F_t) 时间特征
            service_indices: (B*N) 索引
            service_phases: (B, N)
            service_amplitudes: (B, N)
            dec_input_flat: (B*N, P, 1) 解码器输入
        Returns:
            predictions: (B*N, P, Q)
        """

        # 1. 获取服务Embedding
        # (B*N, E_dim)
        e_s = self.service_embedding(service_indices, service_phases, service_amplitudes)

        # 2. 准备编码器输入
        # (B*N, E_dim) -> (B*N, S, E_dim)
        e_s_tiled_enc = e_s.unsqueeze(1).expand(-1, self.seq_len, -1)

        # (B*N, S+P, 1) -> (B*N, S, 1)
        seasonal_enc = seasonal_prior_flat[:, :self.seq_len, :]
        # (B*N, S+P, F_t) -> (B*N, S, F_t)
        time_enc = time_features_flat[:, :self.seq_len, :]

        # (B*N, S, 1+1+F_t+E_dim)
        encoder_input = torch.cat([seq_x_flat, seasonal_enc, time_enc, e_s_tiled_enc], dim=-1)

        # 3. 编码
        enc_out = self.input_projection(encoder_input)  # (B*N, S, D)
        enc_out = self.pos_encoding(enc_out)
        memory = self.transformer_encoder(enc_out)  # (B*N, S, D)

        # 4. 准备解码器输入
        # (B*N, E_dim) -> (B*N, P, E_dim)
        e_s_tiled_dec = e_s.unsqueeze(1).expand(-1, self.pred_len, -1)

        # (B*N, S+P, 1) -> (B*N, P, 1)
        seasonal_dec = seasonal_prior_flat[:, self.seq_len:, :]
        # (B*N, S+P, F_t) -> (B*N, P, F_t)
        time_dec = time_features_flat[:, self.seq_len:, :]

        # (B*N, P, 1+1+F_t+E_dim)
        decoder_input = torch.cat([dec_input_flat, seasonal_dec, time_dec, e_s_tiled_dec], dim=-1)

        # 5. 解码
        dec_out = self.decoder_projection(decoder_input)  # (B*N, P, D)
        dec_out = self.pos_encoding(dec_out)
        dec_out = self.transformer_decoder(dec_out, memory)  # (B*N, P, D)

        # 6. 输出头
        # (B*N, P, Q)
        predictions = self.quantile_head(dec_out)

        return predictions


def monotone_rearrangement(quantile_predictions, quantiles):
    """
    [已重构]
    Args:
        quantile_predictions: (B*N, P, Q)
    """
    sorted_preds, _ = torch.sort(quantile_predictions, dim=2)
    return sorted_preds