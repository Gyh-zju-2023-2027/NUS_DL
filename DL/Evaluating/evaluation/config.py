import torch
import sys
import os
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

data_root_path = './sample_data'


"""
WARNING: if use eval_stage2.py, set FLAG_SENSECOACH = True. If use eval_textonly.py, set FLAG_SENSECOACH = False. If use eval_vllm.py, set FLAG_SENSECOACH = False.
"""
FLAG_SENSECOACH = True

# region ablation study default
# USE_SENSOR_PROCESS = FLAG_BASELINE
USE_FULL_SENSOR_SEQUENCE = True
USE_LSTM_LAYER = False
USE_LEARNABLE_MAPPING = True
# endregion

# models
trained_model_path = r"../models/weights/final_model_6_usinglstm_0_usingfulls_1.pth"
learning_rate = 0.00001
num_epochs = 2
top_k = 6
# endregion
