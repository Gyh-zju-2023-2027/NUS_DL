import torch
import torch.nn as nn

from .key_dim import PREDEFINE_pose_dim_keys
from .setting import USE_FULL_SENSOR_SEQUENCE


class PoseEncoder(nn.Module):
    def __init__(self, pose_input_size, output_dim=64):
        super(PoseEncoder, self).__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(pose_input_size, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        batch_size = x.size(0)
        stroke_num = x.size(1)
        pose_key_dim_nums = x.size(2)
        pose_frames = x.size(3)
        pose_coordinates = x.size(4)
        
        # 将pose数据重塑为 [batch_size * stroke_num * pose_key_dim_nums, pose_frames * pose_coordinates]
        x = x.view(batch_size * stroke_num * pose_key_dim_nums, -1)
        encoded = self.encoder(x)
        output = encoded.view(batch_size, stroke_num, pose_key_dim_nums, -1)
        return output

