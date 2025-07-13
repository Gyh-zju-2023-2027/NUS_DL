# -*- coding: utf-8 -*-

"""
File Name: eval_score.py
Description: 基于训练好的模型权重，对测试数据进行动作评分和建议key生成
"""

import tqdm
import json
import time
from typing import List, Tuple, Set
import torch
import sys
import numpy as np
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from eval_data_loader import get_eval_loader
from models.src.new_key_mapping import allowed_sugg_keys
from load_model import load_model
from models.src.key_dim import  PREDEFINE_pose_dim_keys, RoundDataIncludesPoseSensor
from config import device, top_k, FLAG_SENSECOACH, trained_model_path
from typing import List

def evaluate_suggestion_f1(predicted_key_type: Set[Tuple[str, str]], true_key_type: Set[Tuple[str, str]]):
    """
    F1 score of suggestion_type
    """
    predicted_set = predicted_key_type
    true_set = true_key_type
    precision = len(predicted_set & true_set) / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = len(predicted_set & true_set) / len(true_set) if len(true_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def calc_key(llm_record_output):
    """计算建议key的precision, recall, f1"""
    precision_list_key = []
    recall_list_key = []
    f1_list_key = []
    for round_ in llm_record_output:
        round_sugg_key_set = set([item[0] for item in round_['true_set']])
        sugg_key_set = set([sugg_key[0] for sugg_key in round_['all_llm_pred_sugg_key_types_this_round']])
        if(len(sugg_key_set) != 6):
            # print(round_['round_.round_meta_info'])
            # print(sugg_key_set)  
            pass          
        precission_key, recall_key, f1_key = evaluate_suggestion_f1(sugg_key_set, round_sugg_key_set)
        precision_list_key.append(precission_key)
        recall_list_key.append(recall_key)
        f1_list_key.append(f1_key)
    avg_precision_key = sum(precision_list_key)/len(precision_list_key)
    avg_recall_key = sum(recall_list_key)/len(recall_list_key)
    avg_f1_key = sum(f1_list_key)/len(f1_list_key)
    print("avg_precision_key:",avg_precision_key,"avg_recall_key:",avg_recall_key,"avg_f1_key:",avg_f1_key)
    return avg_precision_key,avg_recall_key,avg_f1_key

def calc_keytype(llm_record_output):
    """计算建议key+type的precision, recall, f1"""
    precision_list_with_no_wrong_key = []
    recall_list_with_no_wrong_key = []
    f1_list_with_no_wrong_key = []
    for round_ in llm_record_output:
        all_llm_pred_sugg_key_types_this_round = set([tuple(item) for item in round_['all_llm_pred_sugg_key_types_this_round']])
        round_sugg_key_type_set = set([tuple(item) for item in round_['true_set']])
        precission_key_type_no_wrong_key, recall_key_type_no_wrong_key, f1_key_type_no_wrong_key = evaluate_suggestion_f1(all_llm_pred_sugg_key_types_this_round, round_sugg_key_type_set)
        precision_list_with_no_wrong_key.append(precission_key_type_no_wrong_key)
        recall_list_with_no_wrong_key.append(recall_key_type_no_wrong_key)
        f1_list_with_no_wrong_key.append(f1_key_type_no_wrong_key)
    avg_precision_key_type_no_wrong_key = sum(precision_list_with_no_wrong_key)/len(precision_list_with_no_wrong_key)
    avg_recall_key_type_no_wrong_key = sum(recall_list_with_no_wrong_key)/len(recall_list_with_no_wrong_key)
    avg_f1_key_type_no_wrong_key = sum(f1_list_with_no_wrong_key)/len(f1_list_with_no_wrong_key)
    print("avg_precision_key_type:",avg_precision_key_type_no_wrong_key,"avg_recall_key_type:",avg_recall_key_type_no_wrong_key,"avg_f1_key_type:",avg_f1_key_type_no_wrong_key)    
    return avg_precision_key_type_no_wrong_key,avg_recall_key_type_no_wrong_key,avg_f1_key_type_no_wrong_key

class RoundDataWithSuggKeys:
    def __init__(self, pred_sugg_keys, true_sugg_keys, sugg_key_type_set, round_meta_info, pose_data, sensor_data, stroke_mask):
        self.pred_sugg_keys = pred_sugg_keys
        self.true_sugg_keys = true_sugg_keys
        self.sugg_key_type_set = sugg_key_type_set
        self.round_meta_info = round_meta_info
        self.pose_data = pose_data
        self.sensor_data = sensor_data
        self.stroke_mask = stroke_mask

def get_sugg_keys(model, test_data_loader, topk=6) -> List[RoundDataWithSuggKeys]:
    """
    从模型输出中获取topk建议keys
    """
    round_data_with_gen_sugg_keys = []
    with torch.no_grad():
        for batch in test_data_loader:
            sensor_x, pose_x, stroke_mask, round_meta_info, sugg_key_type_set, targets = batch
            sensor_x = sensor_x.to(device, non_blocking=True)
            pose_x = pose_x.to(device, non_blocking=True)
            stroke_mask = stroke_mask.to(device, non_blocking=True)
            logits = model(sensor_x, pose_x, stroke_mask)
            probs = torch.sigmoid(logits)
            _, topk_indices = probs.topk(topk, dim=1)
            for batch_idx, batch_sugg_keys in enumerate(topk_indices.tolist()):
                true_keys = []
                for (key,type) in sugg_key_type_set[batch_idx]:
                    true_keys.append(key)
                round_data_with_gen_sugg_keys.append(RoundDataWithSuggKeys(
                    pred_sugg_keys=[allowed_sugg_keys[idx] for idx in batch_sugg_keys],
                    true_sugg_keys=true_keys,
                    round_meta_info=round_meta_info[batch_idx],
                    pose_data=pose_x[batch_idx].tolist(),
                    sensor_data=sensor_x[batch_idx].tolist(),
                    stroke_mask=stroke_mask[batch_idx].tolist(),
                    sugg_key_type_set=sugg_key_type_set[batch_idx]
                ))
    return round_data_with_gen_sugg_keys

def calculate_action_score(probs):
    """
    根据模型输出的概率计算动作评分
    :param probs: 模型输出的概率 [batch, num_keys]
    :return: 动作评分 (0-100分，越高越标准)
    """
    # 方法1: 用平均概率的补数作为评分
    score = 100 * (1 - probs.mean().item())
    return score

def evaluate_action_quality(model, test_loader, topk=6):
    """
    评估动作质量：获取建议keys和评分
    """
    all_results = []
    all_scores = []
    
    with torch.no_grad():
        for pose_x, stroke_mask, meta in test_loader:
            pose_x = pose_x.to(device)
            stroke_mask = stroke_mask.to(device)
            
            logits = model(pose_x, stroke_mask)
            probs = torch.sigmoid(logits)
            _, topk_indices = probs.topk(topk, dim=1)
            
            for batch_idx in range(probs.shape[0]):
                # 获取建议keys
                sugg_keys = [allowed_sugg_keys[idx] for idx in topk_indices[batch_idx].tolist()]
                sugg_scores = probs[batch_idx][topk_indices[batch_idx]].tolist()
                
                # 计算动作评分
                action_score = calculate_action_score(probs[batch_idx])
                all_scores.append(action_score)
                
                # 保存结果 - 处理meta数据格式
                if isinstance(meta, list):
                    meta_data = meta[batch_idx]
                else:
                    meta_data = meta  # 如果是单个字典，直接使用
                
                result = {
                    'meta': meta_data,
                    'suggestion_keys': sugg_keys,
                    'suggestion_scores': sugg_scores,
                    'action_score': action_score
                }
                all_results.append(result)
                
                print(f"\n=== Action Assessment Results ===")
                print(f"Round Info: {result['meta']}")
                print(f"Action Score: {action_score:.1f}/100")
                print(f"Keys that need improvement:")
                for i, (key, score) in enumerate(zip(sugg_keys, sugg_scores)):
                    print(f"  {i+1}. {key}: {score:.3f}")
    
    # 计算统计信息
    if all_scores:
        avg_score = np.mean(all_scores)
        print(f"\n=== Overall Statistics ===")
        print(f"Average Action Score: {avg_score:.1f}/100")
        print(f"Highest Score: {max(all_scores):.1f}")
        print(f"Lowest Score: {min(all_scores):.1f}")
    else:
        print("\n=== Overall Statistics ===")
        print("No valid evaluation data, please check test set content and path.")
    
    return all_results, all_scores

if __name__ == "__main__":
    # 模型参数设置
    fps = 10
    pose_input_size = fps * 3  # 每个关节有fps帧，每帧3个坐标(x,y,z)
    local_model_path = trained_model_path
    test_dir = r"../../../../sampledata/test"  # 按实际路径修改

    # Load model
    print("Loading model...")
    model = load_model(pose_input_size, USE_LSTM_LAYER=False, path=local_model_path, USE_SENSOR_PROCESS=FLAG_SENSECOACH)
    model.eval()

    # Load evaluation data
    print("Loading evaluation data...")
    test_loader = get_eval_loader(test_dir, batch_size=1)
    # Get dataset size to avoid type check errors
    dataset_size = getattr(test_loader.dataset, '__len__', lambda: 'Unknown')()
    print(f"test_loader data count: {dataset_size}")

    # Evaluate
    print("Starting action quality evaluation...")
    results, scores = evaluate_action_quality(model, test_loader, topk=top_k)