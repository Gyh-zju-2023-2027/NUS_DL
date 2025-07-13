import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.src.key_dim import PREDEFINE_pose_dim_keys
from config import FLAG_SENSECOACH

class EvalDataset(Dataset):
    def __init__(self, test_dir):
        self.samples = []
        max_stroke_num = 40
        num_joints = len(PREDEFINE_pose_dim_keys)
        num_coordinates = 3
        fps = 10

        for player_id in os.listdir(test_dir):
            player_dir = os.path.join(test_dir, player_id)
            if not os.path.isdir(player_dir):
                continue
            
                # pose
            pose_path = os.path.join(player_dir, "aligned_downsampled.json")
            try:
                with open(pose_path, 'r') as file:
                    pose_data_json = json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"The file at {pose_path} was not found.")
            except json.JSONDecodeError:
                raise ValueError(f"The file at {pose_path} is not a valid JSON file.")
            extracted_pose_data = []
            num_frames = 10  
            num_joints = len(PREDEFINE_pose_dim_keys)
            num_coordinates = 3 
            for stroke in pose_data_json:
                stroke_pose_data = np.zeros((num_joints, fps, num_coordinates))
                stroke_frame_data = stroke.get("frames", [])
                for frame_index, frame_data in enumerate(stroke_frame_data):
                    if frame_index >= fps:
                        break
                    landmarks = frame_data.get('landmarks', [])
                    for landmark in landmarks:
                        landmark_id, x, y, z = landmark
                        if landmark_id in range(num_joints):
                            stroke_pose_data[landmark_id, frame_index, :] = [x, y, z]
                extracted_pose_data.append(stroke_pose_data)
            
            # mask
            min_len = len(extracted_pose_data)
                    
            if len(extracted_pose_data) < max_stroke_num:
                padding_rows = max_stroke_num - len(extracted_pose_data)
                fps = 10
                padding_data = np.zeros((padding_rows, len(PREDEFINE_pose_dim_keys), fps, 3))
                extracted_pose_data.extend(padding_data.tolist())
                    
            stroke_mask = [1] * min_len + [0] * (max_stroke_num - min_len)
            mask_arr = np.array(stroke_mask, dtype=np.int32)
            
            # ��padding֮�󴴽�����
            pose_arr = np.array(extracted_pose_data, dtype=np.float32)

            self.samples.append({
                "pose": pose_arr,
                "mask": mask_arr,
                "meta": {
                    "player_id": player_id,
                }
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            torch.from_numpy(item["pose"]),
            torch.from_numpy(item["mask"]),
            item["meta"]
        )

def get_eval_loader(test_dir, batch_size=1):
    dataset = EvalDataset(test_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader