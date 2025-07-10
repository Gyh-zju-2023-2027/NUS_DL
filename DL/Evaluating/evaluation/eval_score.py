import torch
from models.src.predict_model import FusionModel
from models.src.new_key_mapping import allowed_sugg_keys
from models.src.setting import device

# 1. 加载模型结构和权重
sensor_input_size = 60   # 示例：你的传感器特征维度
pose_input_size = 90     # 示例：你的姿态特征维度
USE_LSTM_LAYER = True
USE_LEARNABLE_MAPPING = True
USE_SENSOR_PROCESS = True

model = FusionModel(
    sensor_input_size,
    pose_input_size,
    USE_LSTM_LAYER=USE_LSTM_LAYER,
    USE_LEARNABLE_MAPPING=USE_LEARNABLE_MAPPING,
    USE_SENSOR_PROCESS=USE_SENSOR_PROCESS
)
model.load_state_dict(torch.load('models/weights/final_model_6_usinglstm_0_usingfulls_1.pth', map_location=device))
model.eval()

# 2. 准备输入数据（以单个样本为例，实际请用你的数据替换）
# 假设 S=100，sensorK=60，poseK=30
sensor_x = torch.randn(1, 100, 60).to(device)      # [batch, S, sensorK]
pose_x = torch.randn(1, 100, 30).to(device)        # [batch, S, poseK]
stroke_mask = torch.ones(1, 100).to(device)        # [batch, S]

# 3. 推理并获取建议key和分数
with torch.no_grad():
    output = model(sensor_x, pose_x, stroke_mask)  # [batch, key_num]
    probs = torch.sigmoid(output)                  # 概率化
    topk = 6
    topk_probs, topk_indices = probs.topk(topk, dim=1)  # [batch, topk]

# 4. 转为建议key
topk_keys = [allowed_sugg_keys[idx] for idx in topk_indices[0].cpu().numpy()]
print("最需改进的动作建议key：", topk_keys)
print("对应分数（越高越需改进）：", topk_probs[0].cpu().numpy())

# 5. 生成整体动作评分（0-100分，越高越标准）
score = 100 * (1 - probs.mean().item())
print(f"动作标准程度评分（0-100，越高越标准）：{score:.1f}")