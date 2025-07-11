# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def predict_trajectory_lstm(trajectory, future_steps=5, window_size=5):
    """
    Use LSTM model to predict ping-pong trajectory.

    Args:
        trajectory: list of (x, y) coordinates
        future_steps: number of future points to predict
        window_size: length of input window for LSTM

    Returns:
        Draws the real and predicted trajectory, and returns the predicted points array
    """

    # 检查轨迹点数量是否足够
    if len(trajectory) < window_size + 1:
        print("轨迹点太少，至少需要 {} 个点".format(window_size + 1))
        return

    # 1. 数据准备
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

    # 2. 构建模型
    model = Sequential([
        LSTM(64, input_shape=(window_size, 2), return_sequences=False),
        Dense(32, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')

    # 3. 训练模型
    model.fit(X, y, epochs=300, verbose=0)

    # 4. 预测未来轨迹
    current_seq = X[-1]
    predictions = []
    for _ in range(future_steps):
        pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)[0]
        predictions.append(pred)
        current_seq = np.vstack([current_seq[1:], pred])

    # 5. 反归一化 + 可视化
    data[:, 0] *= x_max
    data[:, 1] *= y_max
    predictions = np.array(predictions)
    predictions[:, 0] *= x_max
    predictions[:, 1] *= y_max

    # 绘制真实轨迹和预测轨迹
    plt.figure(figsize=(8, 6))
    plt.plot(data[:, 0], data[:, 1], 'bo-', label='Real Trajectory')
    plt.plot(predictions[:, 0], predictions[:, 1], 'ro--', label='Predicted Trajectory')
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("LSTM Ping-pong Trajectory Prediction")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()
    plt.show()

    return predictions  # 返回预测点数组
