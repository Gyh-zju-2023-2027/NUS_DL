import numpy as np


def extract_features(sensor_window, pose_window, window_size=None):
    """
    从传感器和姿态窗口中提取特征向量。

    参数:
    - sensor_window: 列表或数组，形状为 (T, D1)，T 为时间步，D1 为传感器通道数。
    - pose_window: 列表或数组，形状为 (T, D2)，D2 为姿态特征维度（如关键点坐标）。
    - window_size: 可选，固定窗口长度；若提供，将对序列做截断或零填充。

    返回:
    - features: 一维 numpy 数组，包含时间域与频域特征。
    """
    # 转为 numpy 数组
    sensor_arr = np.array(sensor_window, dtype=float)
    pose_arr = np.array(pose_window, dtype=float)

    # 如需固定长度，则截断或零填充
    if window_size is not None:
        def pad_or_trunc(arr, size):
            T, D = arr.shape
            if T < size:
                pad = np.zeros((size - T, D), dtype=float)
                return np.vstack([arr, pad])
            return arr[:size]

        sensor_arr = pad_or_trunc(sensor_arr, window_size)
        pose_arr = pad_or_trunc(pose_arr, window_size)

    # 1. 时间域特征：均值、标准差、最大值、最小值
    s_mean = sensor_arr.mean(axis=0)
    s_std  = sensor_arr.std(axis=0)
    s_max  = sensor_arr.max(axis=0)
    s_min  = sensor_arr.min(axis=0)

    # 2. 频域特征：对称傅里叶变换后取幅值的均值与标准差
    freq = np.fft.rfft(sensor_arr, axis=0)
    freq_mag = np.abs(freq)
    f_mean = freq_mag.mean(axis=0)
    f_std  = freq_mag.std(axis=0)

    sensor_feat = np.concatenate([s_mean, s_std, s_max, s_min, f_mean, f_std])

    # 3. 姿态特征：均值与标准差
    p_mean = pose_arr.mean(axis=0)
    p_std  = pose_arr.std(axis=0)
    pose_feat = np.concatenate([p_mean, p_std])

    # 合并
    features = np.concatenate([sensor_feat, pose_feat])
    return features


if __name__ == "__main__":
    # 简单测试
    dummy_sensor = [[0.1, -0.2, 0.3]] * 30  # 30 steps, 3 channels
    dummy_pose   = [[0.5] * 33] * 30         # 30 steps, 33-dim pose
    feats = extract_features(dummy_sensor, dummy_pose, window_size=50)
    print("Feature vector length:", feats.shape)
