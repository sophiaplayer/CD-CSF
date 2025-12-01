# seasonality.py
# 季节性分析模块：STL分解、相位对齐、模板生成

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.spatial.distance import euclidean
from scipy.ndimage import uniform_filter1d

try:
    from fastdtw import fastdtw
except ImportError:
    print("警告: fastdtw未安装，Soft-DTW对齐功能将不可用")
    print("可以使用: pip install fastdtw")
    fastdtw = None


class SeasonalityAnalyzer:
    """
    季节性分析器
    实现LaTeX中Section 3.1.1的季节性和相位模板算法
    """

    def __init__(self, period=24, smooth_weight=0.1):
        """
        Args:
            period: 季节性周期长度
            smooth_weight: 指数平滑权重 ω ∈ (0,1)
        """
        self.period = period
        self.smooth_weight = smooth_weight

        # 存储每个服务的季节性参数
        self.service_seasonality = {}  # {service: {'S': seasonal, 'phi': phase, 'a': amplitude}}
        self.shared_template = None  # 共享季节性模板 S̄(t)

    def interpolate_series(self, series, factor=2):
        """
        通过插值增加数据点（相当于减小时隙大小）

        Args:
            series: 原始时间序列
            factor: 插值倍数（2表示数据点增加一倍）

        Returns:
            interpolated_series: 插值后的序列
            original_indices: 原始数据点在新序列中的索引
        """
        from scipy.interpolate import interp1d

        n = len(series)
        x_old = np.arange(n)
        x_new = np.linspace(0, n - 1, n * factor)

        # 使用三次样条插值
        try:
            f = interp1d(x_old, series, kind='cubic', fill_value='extrapolate')
        except:
            # 如果数据太少，使用线性插值
            f = interp1d(x_old, series, kind='linear', fill_value='extrapolate')

        interpolated = f(x_new)

        # 记录原始数据点的索引（每factor个点对应一个原始点）
        original_indices = np.arange(0, len(interpolated), factor)

        return interpolated, original_indices

    def robust_stl_decompose(self, series, period=None):
        """
        使用Robust STL分解时间序列（自动处理数据不足）

        [最终修复] 1. 确保周期(period)总是奇数
        [最终修复] 2. 使用正确的 'period' 关键字参数调用STL
        """
        if period is None:
            period = self.period

        original_series = series.copy()
        original_length = len(series)
        interpolated = False
        interp_factor = 1

        # 检查数据长度是否足够STL（需要至少2*period+1）
        min_required = 2 * period + 1

        # 如果数据不足，进行插值
        if original_length < min_required:
            print(f"    数据量不足 ({original_length} < {min_required})")
            # 计算需要的插值倍数
            interp_factor = int(np.ceil(min_required / original_length)) + 1
            print(f"    通过插值增加数据点 (倍数={interp_factor})，相当于时隙减小到原来的1/{interp_factor}")

            series, original_indices = self.interpolate_series(series, factor=interp_factor)
            interpolated = True
            print(f"    插值后数据点: {len(series)}")

        adjusted_period = period

        # 如果插值后周期变得太大，调整周期
        if len(series) < 3 * adjusted_period:
            adjusted_period = max(3, len(series) // 3)
            print(f"    自动调整周期: {period} -> {adjusted_period}")

        # [修复 1] 奇数检查必须在所有调整之后、调用STL之前执行
        if adjusted_period % 2 == 0:
            adjusted_period += 1
            print(f"    STL周期调整为奇数: {adjusted_period}")

        # 尝试STL分解
        try:
            # 确保周期 < 数据长度，且为奇数
            if adjusted_period >= len(series):
                adjusted_period = max(3, len(series) // 2)
                if adjusted_period % 2 == 0:
                    adjusted_period = max(3, adjusted_period - 1)
                print(f"    周期大于数据长度，强制调整为: {adjusted_period}")

            # [修复 2] 使用 'period' 关键字参数，而不是 'seasonal'
            stl = STL(series, period=adjusted_period, robust=True)
            result = stl.fit()

            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid

            # 如果进行了插值，需要将结果映射回原始长度
            if interpolated:
                from scipy.interpolate import interp1d
                x_new = np.arange(len(trend))
                x_old = np.linspace(0, len(trend) - 1, original_length)

                # 插值回原始长度
                f_trend = interp1d(x_new, trend, kind='linear', fill_value='extrapolate')
                f_seasonal = interp1d(x_new, seasonal, kind='linear', fill_value='extrapolate')
                f_residual = interp1d(x_new, residual, kind='linear', fill_value='extrapolate')

                trend = f_trend(x_old)
                seasonal = f_seasonal(x_old)
                residual = f_residual(x_old)

            print(f"    ✓ STL分解成功")
            return trend, seasonal, residual

        except Exception as e:
            print(f"    STL分解失败 ({e})，使用备用方法")

            # 备用方法：简单的季节性提取
            from scipy.ndimage import uniform_filter1d

            # 使用原始数据
            series = original_series

            # 趋势：移动平均
            window = min(max(3, len(series) // 3), len(series))
            if window % 2 == 0:
                window += 1
            trend = uniform_filter1d(series, size=window, mode='nearest')

            # 季节性：周期平均
            detrended = series - trend
            seasonal = np.zeros_like(series)

            # 使用实际周期（可能比设定的小）
            actual_period = min(period, len(series) // 2)
            if actual_period > 0:
                for i in range(len(series)):
                    phase = i % actual_period
                    indices = [j for j in range(len(series)) if j % actual_period == phase]
                    if indices:
                        seasonal[i] = np.mean(detrended[indices])

            residual = series - trend - seasonal

            print(f"    ✓ 使用备用方法完成分解")
            return trend, seasonal, residual

    def circular_correlation(self, s1, s2, period=None):
        """
        计算循环相关以找到最佳相位偏移

        实现: φ_s = argmax_φ (1/P) Σ_t S_s(t) * S̃((t-φ) mod P)

        Args:
            s1: 第一个季节性序列
            s2: 参考季节性模板
            period: 周期长度

        Returns:
            best_shift: 最佳相位偏移 φ
            max_corr: 最大相关系数
        """
        if period is None:
            period = self.period

        # 提取一个完整周期的数据
        if len(s1) >= period:
            s1_cycle = s1[:period]
        else:
            s1_cycle = np.tile(s1, int(np.ceil(period / len(s1))))[:period]

        if len(s2) >= period:
            s2_cycle = s2[:period]
        else:
            s2_cycle = np.tile(s2, int(np.ceil(period / len(s2))))[:period]

        # 归一化
        s1_norm = (s1_cycle - np.mean(s1_cycle)) / (np.std(s1_cycle) + 1e-8)
        s2_norm = (s2_cycle - np.mean(s2_cycle)) / (np.std(s2_cycle) + 1e-8)

        # 计算所有可能偏移的相关
        correlations = []
        for shift in range(period):
            s2_shifted = np.roll(s2_norm, -shift)
            corr = np.sum(s1_norm * s2_shifted) / period
            correlations.append(corr)

        best_shift = np.argmax(correlations)
        max_corr = correlations[best_shift]

        return best_shift, max_corr

    def soft_dtw_align(self, s1, s2, radius=5):
        """
        使用Soft-DTW进行精细时间规整

        Args:
            s1: 待对齐序列
            s2: 参考序列
            radius: DTW窗口半径

        Returns:
            aligned_s1: 对齐后的s1
            path: DTW路径
        """
        if fastdtw is None:
            # 如果fastdtw未安装，返回原序列
            print("    提示: fastdtw未安装，跳过Soft-DTW对齐")
            return s1, None

        # 使用FastDTW进行快速对齐
        try:
            distance, path = fastdtw(s1.reshape(-1, 1), s2.reshape(-1, 1),
                                     radius=radius, dist=euclidean)

            # 使用路径对s1进行插值对齐
            aligned_indices = [p[0] for p in path]
            aligned_s1 = s1[aligned_indices]

            return aligned_s1, path
        except Exception as e:
            print(f"    警告: Soft-DTW对齐失败 ({e})，返回原序列")
            return s1, None

    def procrustes_normalize(self, curves):
        """
        Procrustes归一化：去除幅度和偏移歧义

        Args:
            curves: list of 1D arrays (每个服务的季节性曲线)

        Returns:
            normalized_curves: 归一化后的曲线列表
        """
        normalized = []
        for curve in curves:
            # 中心化
            centered = curve - np.mean(curve)
            # 归一化到单位范数
            norm = np.linalg.norm(centered)
            if norm > 1e-8:
                normalized.append(centered / norm)
            else:
                normalized.append(centered)

        return normalized

    def aggregate_template(self, aligned_curves):
        """
        聚合对齐后的曲线为共享模板

        实现: S̄(t) = Quantile_0.5({Š_s(t)}_s)

        Args:
            aligned_curves: list of aligned seasonal curves

        Returns:
            template: 中位数模板
        """
        # 确保所有曲线长度一致
        min_len = min(len(c) for c in aligned_curves)
        aligned_matrix = np.array([c[:min_len] for c in aligned_curves])

        # 逐点计算中位数
        template = np.median(aligned_matrix, axis=0)

        return template

    def analyze_multi_service(self, data_dict, reference_service=None):
        """
        分析多个服务的季节性并生成共享模板

        Args:
            data_dict: {service_name: time_series_array}
            reference_service: 参考服务（如果为None则使用第一个）

        Returns:
            results: 包含每个服务的季节性参数和共享模板
        """
        service_names = list(data_dict.keys())

        # Step 1: 对每个服务进行STL分解
        print("  [1/5] 执行STL分解...")
        decompositions = {}
        for service, series in data_dict.items():
            trend, seasonal, residual = self.robust_stl_decompose(series)
            decompositions[service] = {
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual
            }

        # Step 2: 选择参考模板（使用第一个服务或指定服务）
        if reference_service is None:
            reference_service = service_names[0]

        reference_seasonal = decompositions[reference_service]['seasonal']

        # Step 3: 计算每个服务的相位偏移
        print("  [2/5] 计算相位对齐...")
        phases = {}
        for service in service_names:
            seasonal = decompositions[service]['seasonal']
            phi, corr = self.circular_correlation(seasonal, reference_seasonal)
            phases[service] = phi
            print(f"    - {service}: φ = {phi}, 相关系数 = {corr:.4f}")

        # Step 4: 对齐季节性曲线（可选：使用Soft-DTW精细对齐）
        print("  [3/5] 对齐季节性曲线...")
        aligned_curves = []
        for service in service_names:
            seasonal = decompositions[service]['seasonal']
            # 应用相位偏移
            phi = phases[service]
            shifted = np.roll(seasonal, -phi)
            aligned_curves.append(shifted)

        # Step 5: Procrustes归一化
        print("  [4/5] Procrustes归一化...")
        normalized_curves = self.procrustes_normalize(aligned_curves)

        # Step 6: 聚合为共享模板
        print("  [5/5] 生成共享季节性模板...")
        self.shared_template = self.aggregate_template(normalized_curves)

        # 计算每个服务的幅度系数
        # a_s = Median_t {Š_s(t) / S̄(t)}
        amplitudes = {}
        for i, service in enumerate(service_names):
            # 避免除零
            ratio = normalized_curves[i] / (self.shared_template + 1e-8)
            amplitudes[service] = np.median(ratio)

        # 存储结果
        for service in service_names:
            self.service_seasonality[service] = {
                'seasonal': decompositions[service]['seasonal'],
                'trend': decompositions[service]['trend'],
                'residual': decompositions[service]['residual'],
                'phi': phases[service],
                'a': amplitudes[service]
            }

        results = {
            'shared_template': self.shared_template,
            'service_params': self.service_seasonality,
            'period': self.period
        }

        print(f"  ✓ 季节性分析完成")
        print(f"    - 共享模板长度: {len(self.shared_template)}")
        print(f"    - 服务数量: {len(service_names)}")

        return results

    def get_seasonal_prior(self, service, t):
        """
        获取服务s在时间t的季节性先验

        实现: Ŝ_s^prior(t) = a_s * S̄((t + φ_s) mod P)

        Args:
            service: 服务名称
            t: 时间索引（可以是数组）

        Returns:
            prior: 季节性先验值
        """
        if service not in self.service_seasonality:
            return 0.0

        if self.shared_template is None:
            return 0.0

        params = self.service_seasonality[service]
        a = params['a']
        phi = params['phi']

        # 处理数组输入
        if isinstance(t, (list, np.ndarray)):
            t_shifted = (np.array(t) + phi) % self.period
            indices = t_shifted.astype(int) % len(self.shared_template)
            return a * self.shared_template[indices]
        else:
            t_shifted = (t + phi) % self.period
            idx = int(t_shifted) % len(self.shared_template)
            return a * self.shared_template[idx]

    def update_template(self, new_data_dict, update_weight=None):
        """
        指数平滑更新季节性模板

        实现: 每K个周期使用权重ω进行更新

        Args:
            new_data_dict: 新的数据字典
            update_weight: 更新权重（如果为None则使用self.smooth_weight）
        """
        if update_weight is None:
            update_weight = self.smooth_weight

        # 重新分析新数据
        new_results = self.analyze_multi_service(new_data_dict)
        new_template = new_results['shared_template']

        # 指数平滑更新
        if self.shared_template is not None:
            min_len = min(len(self.shared_template), len(new_template))
            self.shared_template[:min_len] = (
                    (1 - update_weight) * self.shared_template[:min_len] +
                    update_weight * new_template[:min_len]
            )
        else:
            self.shared_template = new_template

        print(f"  ✓ 季节性模板已更新 (权重 ω = {update_weight})")


def create_seasonal_features(timestamps, period=24):
    """
    从时间戳创建季节性特征

    Args:
        timestamps: Unix时间戳数组（毫秒）
        period: 周期长度

    Returns:
        features: 包含多种周期特征的数组 [sin, cos, hour, day_of_week, ...]
    """
    # 转换为datetime
    dt = pd.to_datetime(timestamps, unit='ms')

    # 提取时间特征
    hour = dt.hour.values
    day_of_week = dt.dayofweek.values
    day_of_month = dt.day.values

    # 周期性编码（sin/cos变换）
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    features = np.stack([
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        hour / 24,  # 归一化小时
        day_of_week / 7,  # 归一化星期
    ], axis=-1)

    return features