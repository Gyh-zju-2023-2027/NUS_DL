import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def predict_trajectory_lstm(trajectory, future_steps=5, window_size=5):
    """
    使用 LSTM 模型拟合并预测乒乓球轨迹。

    参数：
        trajectory: list of (x, y) 坐标点
        future_steps: 要预测的未来轨迹点数
        window_size: LSTM 输入序列长度

    输出：
        绘图：实际轨迹 + 预测轨迹
    """

    if len(trajectory) < window_size + 1:
        print("轨迹点不足，至少需要 {} 个点".format(window_size + 1))
        return

    # ---------------------
    # 1. 数据准备
    # ---------------------
    data = np.array(trajectory, dtype=np.float32)
    x_max, y_max = data[:, 0].max(), data[:, 1].max()
    data[:, 0] /= x_max
    data[:, 1] /= y_max

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    X = np.array(X)
    y = np.array(y)

    # ---------------------
    # 2. 构建模型
    # ---------------------
    model = Sequential([
        LSTM(64, input_shape=(window_size, 2), return_sequences=False),
        Dense(32, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')

    # ---------------------
    # 3. 训练模型
    # ---------------------
    model.fit(X, y, epochs=300, verbose=0)

    # ---------------------
    # 4. 预测未来轨迹
    # ---------------------
    current_seq = X[-1]
    predictions = []
    for _ in range(future_steps):
        pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)[0]
        predictions.append(pred)
        current_seq = np.vstack([current_seq[1:], pred])

    # ---------------------
    # 5. 反归一化 + 可视化
    # ---------------------
    data[:, 0] *= x_max
    data[:, 1] *= y_max
    predictions = np.array(predictions)
    predictions[:, 0] *= x_max
    predictions[:, 1] *= y_max

    plt.figure(figsize=(8, 6))
    plt.plot(data[:, 0], data[:, 1], 'bo-', label='原始轨迹')
    plt.plot(predictions[:, 0], predictions[:, 1], 'ro--', label='预测轨迹')
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.title("LSTM 乒乓球轨迹预测")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()
    plt.show()

    return predictions  # 可选返回预测点列表
