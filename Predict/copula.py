# copula.py
# 残差依赖建模：t-Copula和联合需求采样

import numpy as np
from scipy import stats
from scipy.stats import t as t_dist
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class ResidualDependenceModeler:
    """
    残差依赖建模器
    实现LaTeX中Section 3.1.3和Algorithm 1
    使用t-Copula建模服务间残差的依赖结构
    """

    def __init__(self, copula_df=5, n_samples=1000):
        """
        Args:
            copula_df: t-Copula的自由度 ν
            n_samples: Monte Carlo采样数量 M
        """
        self.copula_df = copula_df
        self.n_samples = n_samples

        # 存储拟合的参数
        self.correlation_matrix = None  # R_t
        self.marginal_distributions = {}  # {service: marginal_dist}
        self.service_names = None

    def fit_marginal_distributions(self, residuals_dict):
        """
        拟合每个服务残差的边缘分布 F_s

        使用经验CDF + KDE平滑

        Args:
            residuals_dict: {service_name: residual_array}

        Returns:
            marginal_distributions: 拟合的边缘分布字典
        """
        self.service_names = list(residuals_dict.keys())
        self.marginal_distributions = {}

        print("  [Copula 1/3] 拟合边缘分布...")

        for service, residuals in residuals_dict.items():
            # 移除NaN和Inf
            residuals = residuals[np.isfinite(residuals)]

            if len(residuals) < 10:
                print(f"    警告: {service} 残差数据不足，跳过")
                continue

            # 使用KDE拟合（也可以使用参数分布如t分布）
            try:
                kde = stats.gaussian_kde(residuals, bw_method='scott')

                # 存储KDE和数据统计
                self.marginal_distributions[service] = {
                    'type': 'kde',
                    'kde': kde,
                    'data': residuals,
                    'mean': np.mean(residuals),
                    'std': np.std(residuals),
                    'min': np.min(residuals),
                    'max': np.max(residuals),
                    'quantiles': np.percentile(residuals, [1, 5, 25, 50, 75, 95, 99])
                }

                print(f"    - {service}: KDE拟合完成 (n={len(residuals)})")
            except Exception as e:
                print(f"    警告: {service} KDE拟合失败: {e}")
                # 备选：使用经验分布
                self.marginal_distributions[service] = {
                    'type': 'empirical',
                    'data': residuals
                }

        return self.marginal_distributions

    def compute_correlation_matrix(self, residuals_dict, method='kendall'):
        """
        计算残差之间的相关矩阵

        使用Kendall's τ更稳健

        Args:
            residuals_dict: {service_name: residual_array}
            method: 'kendall', 'spearman', or 'pearson'

        Returns:
            correlation_matrix: 相关矩阵 R_t
        """
        print(f"  [Copula 2/3] 计算相关矩阵 (方法: {method})...")

        # 构建残差矩阵 (n_samples × n_services)
        service_names = list(residuals_dict.keys())
        n_services = len(service_names)

        # 获取最短序列长度
        min_len = min(len(v) for v in residuals_dict.values())

        residual_matrix = np.zeros((min_len, n_services))
        for i, service in enumerate(service_names):
            residual_matrix[:, i] = residuals_dict[service][:min_len]

        # 计算相关矩阵
        if method == 'kendall':
            # Kendall's tau
            corr_matrix = np.eye(n_services)
            for i in range(n_services):
                for j in range(i + 1, n_services):
                    tau, _ = stats.kendalltau(residual_matrix[:, i], residual_matrix[:, j])
                    corr_matrix[i, j] = tau
                    corr_matrix[j, i] = tau
        elif method == 'spearman':
            corr_matrix = np.corrcoef(
                stats.rankdata(residual_matrix, axis=0).T
            )
        else:  # pearson
            corr_matrix = np.corrcoef(residual_matrix.T)

        # 确保正定性
        corr_matrix = self._nearest_positive_definite(corr_matrix)

        self.correlation_matrix = corr_matrix

        print(f"    - 相关矩阵形状: {corr_matrix.shape}")
        print(f"    - 平均相关系数: {np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])):.4f}")

        return corr_matrix

    def _nearest_positive_definite(self, A):
        """
        找到最近的正定矩阵
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if self._is_positive_definite(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        return A3

    def _is_positive_definite(self, A):
        """检查矩阵是否正定"""
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    def sample_copula_uniform(self, n_samples=None):
        """
        从t-Copula采样，返回均匀分布的样本

        实现: U^(m) ~ t-Copula(R_t, ν)

        Args:
            n_samples: 采样数量（如果为None则使用self.n_samples）

        Returns:
            uniform_samples: (n_samples, n_services) 的均匀分布样本
        """
        if n_samples is None:
            n_samples = self.n_samples

        if self.correlation_matrix is None:
            raise ValueError("请先调用 fit() 方法")

        n_services = len(self.service_names)

        # Step 1: 从多元t分布采样
        # X ~ t_ν(0, R_t)
        mean = np.zeros(n_services)

        # 多元t分布采样
        samples_t = stats.multivariate_t(
            loc=mean,
            shape=self.correlation_matrix,
            df=self.copula_df
        ).rvs(size=n_samples)

        if n_samples == 1:
            samples_t = samples_t.reshape(1, -1)

        # Step 2: 转换到均匀分布 U = F_t(X)
        uniform_samples = t_dist.cdf(samples_t, df=self.copula_df)

        return uniform_samples

    def inverse_transform_marginal(self, uniform_samples, service):
        """
        将均匀分布样本转换回原始残差空间

        实现: ε̃ = F_s^{-1}(U)

        Args:
            uniform_samples: 均匀分布样本 [0,1]
            service: 服务名称

        Returns:
            residual_samples: 残差空间的样本
        """
        if service not in self.marginal_distributions:
            raise ValueError(f"未找到服务 {service} 的边缘分布")

        marginal = self.marginal_distributions[service]

        if marginal['type'] == 'kde':
            # 使用逆变换采样
            # 简化版：使用经验分位数
            data = marginal['data']
            quantiles = np.quantile(data, uniform_samples)
            return quantiles
        else:
            # 经验分布
            data = marginal['data']
            indices = (uniform_samples * (len(data) - 1)).astype(int)
            return np.sort(data)[indices]

    def fit(self, residuals_dict):
        """
        拟合完整的t-Copula模型

        Args:
            residuals_dict: {service_name: residual_array}

        Returns:
            self
        """
        # 拟合边缘分布
        self.fit_marginal_distributions(residuals_dict)

        # 计算相关矩阵
        self.compute_correlation_matrix(residuals_dict, method='kendall')

        print("  [Copula 3/3] t-Copula拟合完成")

        return self

    def sample_joint_demand(self, median_forecasts, time_steps=1):
        """
        生成联合需求样本并计算聚合需求分位数

        实现Algorithm 1: Joint Residual Sampling and Aggregated Demand Quantile

        Args:
            median_forecasts: {service: median_forecast_array} (中位数预测)
            time_steps: 预测时间步数

        Returns:
            aggregated_quantiles: 聚合需求的分位数 {quantile: values}
            service_samples: 每个服务的样本 {service: samples}
        """
        print(f"  [Copula] 生成联合需求样本 (M={self.n_samples})...")

        if self.correlation_matrix is None:
            raise ValueError("请先调用 fit() 方法")

        # Step 1: 从Copula采样均匀分布
        uniform_samples = self.sample_copula_uniform(self.n_samples)

        # Step 2: 对每个服务转换到残差空间
        service_samples = {}
        for i, service in enumerate(self.service_names):
            if service not in median_forecasts:
                continue

            # 获取该服务的均匀样本
            u_samples = uniform_samples[:, i]

            # 转换到残差空间: ε̃ = F_s^{-1}(U)
            residual_samples = self.inverse_transform_marginal(u_samples, service)

            # 添加到中位数预测: ỹ = q_0.5 + ε̃
            forecast = median_forecasts[service]
            if isinstance(forecast, (list, np.ndarray)):
                # 如果预测是序列，重复采样
                forecast = np.mean(forecast)  # 简化：使用平均值

            demand_samples = forecast + residual_samples
            # 确保非负
            demand_samples = np.maximum(demand_samples, 0)

            service_samples[service] = demand_samples

        # Step 3: 聚合所有服务的需求
        # D^(m) = Σ_s ỹ_s^(m)
        aggregated_samples = np.sum(
            [samples for samples in service_samples.values()],
            axis=0
        )

        # Step 4: 计算聚合需求分位数
        quantile_levels = [0.5, 0.75, 0.9, 0.95, 0.99]
        aggregated_quantiles = {
            q: np.quantile(aggregated_samples, q)
            for q in quantile_levels
        }

        print(f"    - 生成了 {len(service_samples)} 个服务的样本")
        print(f"    - 聚合需求分位数:")
        for q, val in aggregated_quantiles.items():
            print(f"      Q_{q}: {val:.2f}")

        return aggregated_quantiles, service_samples

    def update_correlation(self, new_residuals_dict, smooth_weight=0.1):
        """
        使用指数平滑更新相关矩阵

        实现: 使用Kendall's τ的指数平滑

        Args:
            new_residuals_dict: 新的残差数据
            smooth_weight: 平滑权重
        """
        # 计算新的相关矩阵
        new_corr = self.compute_correlation_matrix(new_residuals_dict, method='kendall')

        # 指数平滑更新
        if self.correlation_matrix is not None:
            self.correlation_matrix = (
                    (1 - smooth_weight) * self.correlation_matrix +
                    smooth_weight * new_corr
            )
        else:
            self.correlation_matrix = new_corr

        print(f"  ✓ 相关矩阵已更新 (权重 = {smooth_weight})")