import os
import json
import argparse
import numpy as np
import torch
from feature_extraction import extract_features
from scoring_model import ScoringModel


def infer_round(round_path, model, window_size, device):
    """
    对指定轮次目录中的所有击球窗口进行评分。
    返回字典: {stroke_index: score}
    """
    sensor_file = os.path.join(round_path, 'sensor', 'matched.json')
    pose_file   = os.path.join(round_path, 'pose',   'matched.json')
    if not (os.path.exists(sensor_file) and os.path.exists(pose_file)):
        return {}

    with open(sensor_file, 'r') as f:
        sensor_data = json.load(f)
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)

    stroke_indices = sorted(sensor_data.keys(), key=lambda x: int(x))
    feats_list = []
    for idx in stroke_indices:
        sensor_window = sensor_data.get(idx)
        pose_window   = pose_data.get(idx)
        if sensor_window is None or pose_window is None:
            continue
        feats = extract_features(sensor_window, pose_window, window_size=window_size)
        feats_list.append(feats)

    if not feats_list:
        return {}

    X = torch.tensor(np.stack(feats_list), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy()

    results = {idx: float(score) for idx, score in zip(stroke_indices, preds)}
    return results


def main():
    parser = argparse.ArgumentParser(description='Run inference for scoring model')
    parser.add_argument('--data_root',   type=str,  required=True, help='Root directory of aligned rounds')
    parser.add_argument('--window_size', type=int,  default=50,    help='Window size for feature extraction')
    parser.add_argument('--model_path',  type=str,  required=True, help='Path to trained model .pth file')
    parser.add_argument('--output_dir',  type=str,  default='results/scores', help='Directory to save score JSONs')
    parser.add_argument('--hidden_dims', type=int,  nargs='+', default=[128,64], help='Hidden layer dims used in training')
    parser.add_argument('--device',      type=str,  default=None, help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(args.output_dir, exist_ok=True)

    # Infer input_dim from one example
    # 找到首个有效窗口
    first_feats = None
    for rn in sorted(os.listdir(args.data_root)):
        rp = os.path.join(args.data_root, rn)
        sf = os.path.join(rp, 'sensor', 'matched.json')
        pf = os.path.join(rp, 'pose',   'matched.json')
        if os.path.isdir(rp) and os.path.exists(sf) and os.path.exists(pf):
            sd = json.load(open(sf))
            pd = json.load(open(pf))
            for idx in sd:
                if idx in pd:
                    first_feats = extract_features(sd[idx], pd[idx], window_size=args.window_size)
                    break
        if first_feats is not None:
            break

    if first_feats is None:
        print("No valid stroke windows found in data_root.")
        return

    input_dim = len(first_feats)
    model = ScoringModel(input_dim=input_dim, hidden_dims=args.hidden_dims).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 对每个轮次进行推理并保存
    for round_name in sorted(os.listdir(args.data_root)):
        round_path = os.path.join(args.data_root, round_name)
        if not os.path.isdir(round_path):
            continue
        results = infer_round(round_path, model, args.window_size, device)
        if not results:
            continue
        out_file = os.path.join(args.output_dir, f"{round_name}_scores.json")
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved scores for {round_name} to {out_file}")


if __name__ == '__main__':
    main()
