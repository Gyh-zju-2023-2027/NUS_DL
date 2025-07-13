import sys
from typing import Tuple, Optional, List, Union, Literal, Set, Dict, get_args
import re
import numpy as np
from typing import Tuple, Optional, List, Union, Literal
import torch
from sklearn.model_selection import train_test_split
import os
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from new_key_mapping import suggKey_to_dataKey_mapping, clean_key_mapping, AllowedSuggestionKey, \
    allowed_sugg_keys, AllowedSuggestionType
from key_dim import RoundExpertSuggestions, StrokeExpertSuggestion, RoundDataIncludesPoseSensor, \
    PREDEFINE_pose_dim_keys, all_data_keys, ParsedSuggestion

sample_case_path = '../sampledata'

def custom_collate_fn(batch):
    inputs, labels = zip(*batch)

    pose_data, stroke_mask_tensor = zip(*inputs)
    pose_data = torch.stack(pose_data)  
    stroke_mask_tensor = torch.stack(stroke_mask_tensor)
    
    round_meta_info, sugg_key_type_set, expert_sugg_key_id_set = zip(*labels)
    round_meta_info = list(round_meta_info)
    sugg_key_type_set = list(sugg_key_type_set)
    expert_sugg_key_id_set = torch.stack(expert_sugg_key_id_set)

    return pose_data, stroke_mask_tensor, round_meta_info, sugg_key_type_set, expert_sugg_key_id_set

def load_and_extract_sugg_from_object(sugg_object: List[Dict]) -> Tuple[List[ParsedSuggestion],List[AllowedSuggestionKey]]:
    parsed_suggestions = []
    suggestion_keys = []
    for sugg in sugg_object:
        key = sugg.get('key', 'others')
        suggestion_type = sugg.get('suggestion_type', 'others')
        description = sugg.get('description', '')
        
        # 清理和验证key
        if key in clean_key_mapping:
            key = clean_key_mapping[key]
        else:
            key = 'others'
            
        parsed_suggestion = ParsedSuggestion(
            key=key,
            suggestion_type=suggestion_type,
            description=description
        )
        parsed_suggestions.append(parsed_suggestion)
        suggestion_keys.append(key)
    
    return parsed_suggestions, suggestion_keys

def load_and_extract_sugg_from_object_stroke(sugg_object: List[Dict]) -> Tuple[List[StrokeExpertSuggestion],List[AllowedSuggestionKey]]:
    stroke_suggestions = []
    suggestion_keys = []
    for sugg in sugg_object:
        stroke_id = sugg.get('stroke_id', 0)
        key = sugg.get('key', 'others')
        suggestion_type = sugg.get('suggestion_type', 'others')
        description = sugg.get('description', '')
        
        # 清理和验证key
        if key in clean_key_mapping:
            key = clean_key_mapping[key]
        else:
            key = 'others'
            
        parsed_suggestion = ParsedSuggestion(
            key=key,
            suggestion_type=suggestion_type,
            description=description
        )
        
        stroke_suggestion = StrokeExpertSuggestion(
            stroke_id=stroke_id,
            suggestion=parsed_suggestion
        )
        stroke_suggestions.append(stroke_suggestion)
        suggestion_keys.append(key)
    
    return stroke_suggestions, suggestion_keys

def load_expert_suggestions(suggestion_path) -> List[RoundExpertSuggestions]:
    suggestions = []
    if os.path.exists(suggestion_path):
        for file_name in os.listdir(suggestion_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(suggestion_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        stroke_suggestions, stroke_keys = load_and_extract_sugg_from_object_stroke(data.get('stroke_suggestions', []))
                        summary_suggestions, summary_keys = load_and_extract_sugg_from_object(data.get('summary_suggestions', []))
                        
                        suggestion = RoundExpertSuggestions(
                            round_meta_info=data.get('round_meta_info', ''),
                            stroke_suggestions=stroke_suggestions,
                            summary_suggestions=summary_suggestions
                        )
                        suggestions.append(suggestion)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return suggestions

def calculate_frequencies(keys_list):
    
    result_dict = {}
    total_keys = len(keys_list)
    
    for key in keys_list:
        if key in result_dict:
            result_dict[key]['count'] += 1
        else:
            result_dict[key] = {'count': 1}

   
    sorted_keys = sorted(result_dict.items(), key=lambda item: item[1]['count'], reverse=True)

    
    print(f"All word frequency information in {total_keys}:")
    
    for i, (key, info) in enumerate(sorted_keys):
        print(f"{i + 1}. {key}: \t\t\tOccurrence count = {info['count']}")


def load_round_data(data_dir=sample_case_path) -> List[RoundDataIncludesPoseSensor]:
    round_data_list:List[RoundDataIncludesPoseSensor] = []
    round_meta_info = {"player_id": "-1", "tech_name": "-1", "round_id": "-1"}
    for player_id in os.listdir(data_dir):
        
        round_meta_info['player_id'] = player_id
        player_dir = os.path.join(data_dir, player_id)
        if os.path.isdir(player_dir):
            for tech_name in os.listdir(player_dir):
                round_meta_info['tech_name'] = tech_name
                tech_dir = os.path.join(player_dir, tech_name)
                if os.path.isdir(tech_dir):
                    for round_id in ['round_00', 'round_01', 'round_02']:
                        round_meta_info['round_id'] = round_id
                        round_dir = os.path.join(tech_dir, round_id)
                        if os.path.isdir(round_dir):
                            
                            pose_dir = os.path.join(round_dir, "pose/matched.json")
                            try:
                                with open(pose_dir, 'r',encoding="utf-8") as file:
                                    pose_data_json = json.load(file)
                            except FileNotFoundError:
                                raise FileNotFoundError(f"The file at {pose_dir} was not found.")
                            except json.JSONDecodeError:
                                raise ValueError(f"The file at {pose_dir} is not a valid JSON file.")
                            
                            extracted_pose_data = []
                            num_frames = 10  
                            num_joints = len(PREDEFINE_pose_dim_keys)
                            num_coordinates = 3  
                            for stroke in pose_data_json:
                                stroke_pose_data = np.zeros((num_joints, num_frames, num_coordinates))
                                stroke_frame_data = stroke.get("frames")
                                for frame_index, frame_data in enumerate(stroke_frame_data):
                                    if frame_index >= num_frames: 
                                        break
                                    landmarks = frame_data.get('landmarks', [])
                                    for landmark in landmarks:
                                        landmark_id, x, y, z = landmark
                                        if landmark_id in range(num_joints):
                                            stroke_pose_data[landmark_id, frame_index, :] = [x, y, z]
                                extracted_pose_data.append(stroke_pose_data)
                           
                            expert_sugg_keys_set = set()
                            suggestion_dir = os.path.join(round_dir, "comments")
                            suggestions: List[RoundExpertSuggestions] = load_expert_suggestions(suggestion_dir)

                            for sugg in suggestions:
                                for stroke_sugg in sugg.stroke_suggestions:
                                    expert_sugg_keys_set.add(allowed_sugg_keys.index(stroke_sugg.suggestion.key))
                                for summary_sugg in sugg.summary_suggestions:
                                    expert_sugg_keys_set.add(allowed_sugg_keys.index(summary_sugg.key))
                            
                            min_len = len(extracted_pose_data)
                            max_stroke_num = 40
                            
                            if len(extracted_pose_data) < max_stroke_num:
                                padding_rows = max_stroke_num - len(extracted_pose_data)
                                fps = 10
                                padding_data = np.zeros((padding_rows, len(PREDEFINE_pose_dim_keys), fps, 3))
                                extracted_pose_data.extend(padding_data.tolist())
                           
                            stroke_mask = [1] * min_len + [0] * (max_stroke_num - min_len)
                            
                            round_data = RoundDataIncludesPoseSensor(
                                round_meta_info='-'.join(round_meta_info.values()),
                                pose_data=extracted_pose_data,
                                stroke_mask=stroke_mask,
                                expert_suggestions=suggestions,
                                expert_sugg_key_id_set=expert_sugg_keys_set
                            )
                            round_data_list.append(round_data)
    print('finished loading')
    return round_data_list

def preprocess_data_as_data_loader(round_data_list_: List[RoundDataIncludesPoseSensor], batch_size: int, fps: int):
    print('len round_data_list:', len(round_data_list_))
    train_val_data, test_data = train_test_split(round_data_list_, test_size=0.2, random_state=42)
    train_data, valid_data = train_test_split(train_val_data, test_size=0.25, random_state=42) 

    def process_data(data_list: List[RoundDataIncludesPoseSensor]):
        pose_list = []
        mask_list = []
        round_meta_info_list = []
        sugg_key_type_set_list = []
        label_matrix = np.zeros((len(data_list), len(allowed_sugg_keys)), dtype=np.float32)
        for idx, data in enumerate(data_list): 
            
            pose_np = np.array(data.pose_data, dtype=np.float32)
            mask_np = np.array(data.stroke_mask, dtype=np.int32)
            all_sugg_types_from_exp: Set[Tuple[AllowedSuggestionKey, AllowedSuggestionType]] = set()
            
            for round_exp_sugs in data.expert_suggestions:
                for stroke_exp_sug in round_exp_sugs.stroke_suggestions:
                    sugg_type = stroke_exp_sug.suggestion.suggestion_type
                    sugg_key = stroke_exp_sug.suggestion.key
                    all_sugg_types_from_exp.add((sugg_key, sugg_type))
                for summary_exp_sug in round_exp_sugs.summary_suggestions:
                    sugg_type = summary_exp_sug.suggestion_type
                    sugg_key = summary_exp_sug.key
                    all_sugg_types_from_exp.add((sugg_key, sugg_type))

            pose_np = pose_np.reshape(-1, len(PREDEFINE_pose_dim_keys), fps, 3)

            pose_list.append(pose_np)
            mask_list.append(mask_np)
            round_meta_info_list.append(data.round_meta_info)
            sugg_key_type_set_list.append(all_sugg_types_from_exp)
            
            if 24 in data.expert_sugg_key_id_set and len(data.expert_sugg_key_id_set) > 1: 
                data.expert_sugg_key_id_set.remove(24)
            label_matrix[idx, [key_id for key_id in data.expert_sugg_key_id_set]] = 1.0
        pose_tensors = [torch.from_numpy(x) for x in pose_list]
        mask_tensors = [torch.from_numpy(x) for x in mask_list]
        label_tensor = torch.from_numpy(label_matrix)

        return list(zip(pose_tensors, mask_tensors)), list(zip(round_meta_info_list, sugg_key_type_set_list, label_tensor))

    train_inputs, train_labels = process_data(train_data)
    valid_inputs, valid_labels = process_data(valid_data)
    test_inputs, test_labels = process_data(test_data)

    class RoundPoseSensorDataset(Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return self.inputs[idx], self.labels[idx]

    train_dataset = RoundPoseSensorDataset(train_inputs, train_labels)
    valid_dataset = RoundPoseSensorDataset(valid_inputs, valid_labels)
    test_dataset = RoundPoseSensorDataset(test_inputs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate_fn, pin_memory=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True, num_workers=8)
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    load_round_data(sample_case_path)