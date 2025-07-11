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

from data_loader import load_round_data, sample_case_path, preprocess_data_as_data_loader
from models.src.new_key_mapping import allowed_sugg_keys, suggKey_to_dataKey_mapping, AllowedSuggestionKey, AllowedSuggestionType
from load_model import load_model
from models.src.key_dim import PREDEFINE_sensor_dim_keys, PREDEFINE_pose_dim_keys, RoundDataIncludesPoseSensor
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
            print(round_['round_.round_meta_info'])
            print(sugg_key_set)            
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

def evaluate_action_quality(model, test_data_loader, topk=6):
    """
    评估动作质量：获取建议keys和评分
    """
    all_results = []
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_data_loader, desc="评估动作质量"):
            sensor_x, pose_x, stroke_mask, round_meta_info, sugg_key_type_set, targets = batch
            sensor_x = sensor_x.to(device, non_blocking=True)
            pose_x = pose_x.to(device, non_blocking=True)
            stroke_mask = stroke_mask.to(device, non_blocking=True)
            
            logits = model(sensor_x, pose_x, stroke_mask)
            probs = torch.sigmoid(logits)
            _, topk_indices = probs.topk(topk, dim=1)
            
            for batch_idx in range(probs.shape[0]):
                # 获取建议keys
                sugg_keys = [allowed_sugg_keys[idx] for idx in topk_indices[batch_idx].tolist()]
                sugg_scores = probs[batch_idx][topk_indices[batch_idx]].tolist()
                
                # 计算动作评分
                action_score = calculate_action_score(probs[batch_idx])
                all_scores.append(action_score)
                
                # 保存结果
                result = {
                    'round_meta_info': round_meta_info[batch_idx],
                    'suggestion_keys': sugg_keys,
                    'suggestion_scores': sugg_scores,
                    'action_score': action_score,
                    'true_sugg_keys': [key for key, _ in sugg_key_type_set[batch_idx]]
                }
                all_results.append(result)
                
                print(f"\n=== 动作评估结果 ===")
                print(f"轮次信息: {result['round_meta_info']}")
                print(f"动作评分: {action_score:.1f}/100")
                print(f"最需改进的建议keys:")
                for i, (key, score) in enumerate(zip(sugg_keys, sugg_scores)):
                    print(f"  {i+1}. {key}: {score:.3f}")
    
    # 计算统计信息
    avg_score = np.mean(all_scores)
    print(f"\n=== 总体统计 ===")
    print(f"平均动作评分: {avg_score:.1f}/100")
    print(f"最高评分: {max(all_scores):.1f}")
    print(f"最低评分: {min(all_scores):.1f}")
    
    return all_results, all_scores

if __name__ == "__main__":
    # 模型参数设置
    sensor_input_size = len(PREDEFINE_sensor_dim_keys)
    fps = 10
    use_lstm = False
    pose_input_size = fps * len(PREDEFINE_pose_dim_keys) * 3 
    num_keys = len(PREDEFINE_sensor_dim_keys) + len(PREDEFINE_pose_dim_keys)
    local_model_path = trained_model_path
    
    # 加载模型
    print("正在加载模型...")
    model = load_model(sensor_input_size, pose_input_size, USE_LSTM_LAYER=use_lstm, path=local_model_path, USE_SENSOR_PROCESS=FLAG_SENSECOACH)
    model.eval()
    
    # 加载测试数据
    print("正在加载测试数据...")
    batch_size = 5
    round_data_list: List[RoundDataIncludesPoseSensor] = load_round_data(sample_case_path)
    train_set, valid_set, test_set = preprocess_data_as_data_loader(round_data_list, batch_size, fps)
    
    # 评估动作质量
    print("开始评估动作质量...")
    results, scores = evaluate_action_quality(model, test_set, topk=top_k)
    
    # 保存结果
    output_file = "action_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存到: {output_file}")
    
    # 计算评估指标（如果有真实标签）
    print("\n=== 评估指标 ===")
    round_data_with_gen_sugg_keys = get_sugg_keys(model, test_set, topk=top_k)
    
    # 准备评估数据
    eval_data = []
    for round_ in round_data_with_gen_sugg_keys:
        all_pred_sugg_key_types_this_round = set()
        for sugg_key in round_.pred_sugg_keys:
            # 这里简化处理，实际可能需要根据具体需求设置suggestion_type
            all_pred_sugg_key_types_this_round.add((sugg_key, "position"))
        
        true_key_type_set = set([tuple(item) for item in round_.sugg_key_type_set])
        
        eval_data.append({
            "all_llm_pred_sugg_key_types_this_round": list(all_pred_sugg_key_types_this_round),
            "true_set": list(true_key_type_set),
            "round_.round_meta_info": round_.round_meta_info
        })
    
    # 计算指标
    key_p, key_r, key_f1 = calc_key(eval_data)
    keytype_p, keytype_r, keytype_f1 = calc_keytype(eval_data)
    
    print(f"建议Key评估 - Precision: {key_p:.4f}, Recall: {key_r:.4f}, F1: {key_f1:.4f}")
    print(f"建议Key+Type评估 - Precision: {keytype_p:.4f}, Recall: {keytype_r:.4f}, F1: {keytype_f1:.4f}")