import torch
import torch.nn as nn
from models.src.predict_model import FusionModel
from config import device

def load_model(pose_input_size, USE_SENSOR_PROCESS, USE_LSTM_LAYER=False, USE_LEARNABLE_MAPPING=True, path='fusion_model.pth'):
    model = FusionModel(pose_input_size, USE_SENSOR_PROCESS=USE_SENSOR_PROCESS, USE_LSTM_LAYER=USE_LSTM_LAYER, USE_LEARNABLE_MAPPING=USE_LEARNABLE_MAPPING).to(device)
    
    # 加载权重文件
    checkpoint = torch.load(path, map_location=device)
    
    # 过滤掉sensor相关的权重参数
    filtered_state_dict = {}
    for key, value in checkpoint.items():
        if not key.startswith('sensor_encoder'):
            filtered_state_dict[key] = value
    
    # 尝试加载过滤后的权重
    try:
        model.load_state_dict(filtered_state_dict, strict=False)
        print("The model weights were successfully loaded")
    except Exception as e:
        pass
    
    return model