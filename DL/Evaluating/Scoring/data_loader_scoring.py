import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

from feature_extraction import extract_features


def collate_fn(batch):
    """
    Collate function to batch feature vectors and scores.
    """
    features = [item[0] for item in batch]
    scores = [item[1] for item in batch]
    # Stack into tensors
    X = torch.stack(features, dim=0)
    y = torch.stack(scores, dim=0)
    return X, y


class ScoringDataset(Dataset):
    """
    Dataset for loading per-stroke feature windows and corresponding scores.
    Expects directory structure:
      root_dir/
        <round_id>/
          sensor/matched.json
          pose/matched.json
          score.json      # List of {"stroke_index": int, "score": float}
    """

    def __init__(self, root_dir, window_size, transform=None):
        self.root_dir = root_dir
        self.window_size = window_size
        self.transform = transform
        self.samples = []

        # Scan all round directories
        for round_name in sorted(os.listdir(self.root_dir)):
            round_path = os.path.join(self.root_dir, round_name)
            if not os.path.isdir(round_path):
                continue

            # Load aligned sensor and pose windows
            sensor_file = os.path.join(round_path, 'sensor', 'matched.json')
            pose_file = os.path.join(round_path, 'pose', 'matched.json')
            score_file = os.path.join(round_path, 'score.json')

            if not (os.path.exists(sensor_file) and os.path.exists(pose_file) and os.path.exists(score_file)):
                continue

            with open(sensor_file, 'r') as f:
                sensor_data = json.load(f)
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
            with open(score_file, 'r') as f:
                score_entries = json.load(f)

            # For each scored stroke, collect its data window
            for entry in score_entries:
                idx = entry['stroke_index']
                score_val = entry['score']

                # Fetch the sensor and pose windows for this stroke
                sensor_window = sensor_data.get(str(idx))
                pose_window = pose_data.get(str(idx))
                if sensor_window is None or pose_window is None:
                    continue

                # Convert raw windows into feature vectors
                features = extract_features(sensor_window, pose_window, window_size=self.window_size)
                features = torch.tensor(features, dtype=torch.float)
                score_tensor = torch.tensor(score_val, dtype=torch.float)

                self.samples.append((features, score_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, score = self.samples[idx]
        if self.transform:
            features = self.transform(features)
        return features, score


def get_data_loaders(root_dir, window_size, batch_size, splits=(0.6, 0.2, 0.2), num_workers=4, shuffle=True):
    """
    Utility to create train/val/test DataLoaders for scoring.
    """
    dataset = ScoringDataset(root_dir, window_size)
    total = len(dataset)
    n_train = int(splits[0] * total)
    n_val = int(splits[1] * total)
    n_test = total - n_train - n_val

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                               num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare DataLoaders for scoring model training.")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing round subfolders')
    parser.add_argument('--window_size', type=int, default=50, help='Fixed window size for feature extraction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_data_loaders(
        root_dir=args.data_root,
        window_size=args.window_size,
        batch_size=args.batch_size
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
