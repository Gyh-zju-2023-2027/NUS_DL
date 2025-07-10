import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ======== Dataset Definition ========
class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, seq_len=10, pred_len=5):
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = []

        coords = self.data[['x', 'y']].values.astype(np.float32)
        for i in range(len(coords) - seq_len - pred_len):
            input_seq = coords[i:i+seq_len]
            target_seq = coords[i+seq_len:i+seq_len+pred_len]
            self.samples.append((input_seq, target_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)

# ======== Model Definition ========
class LSTM_Predictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_len=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len * input_size)
        self.output_len = output_len
        self.input_size = input_size

    def forward(self, x):
        batch_size = x.size(0)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.view(batch_size, self.output_len, self.input_size)

# ======== Training Script ========
def train_model(csv_path="Trajectory_data/trajectory_data.csv", seq_len=10, pred_len=5, epochs=50):
    dataset = TrajectoryDataset(csv_path, seq_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTM_Predictor(output_len=pred_len)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "lstm_model.pth")
    print("Model saved as lstm_model.pth")

if __name__ == "__main__":
    train_model()