import torch
import pandas as pd
import numpy as np
from LSTM_train import LSTM_Predictor

def predict_next(csv_file, model_path='lstm_model.pth', seq_len=10, pred_len=5):
    """
    Load trajectory data from a CSV, feed the last `seq_len` points into the trained LSTM,
    and predict the next `pred_len` (x, y) coordinates.
    """
    # 1. Read data
    df = pd.read_csv(csv_file)
    coords = df[['x', 'y']].values.astype(np.float32)
    if len(coords) < seq_len:
        raise ValueError(f"Not enough data points: need at least {seq_len}, got {len(coords)}")

    # 2. Prepare input sequence
    input_seq = coords[-seq_len:]
    input_tensor = torch.tensor(input_seq).unsqueeze(0)  # shape: (1, seq_len, 2)

    # 3. Load model
    model = LSTM_Predictor(input_size=2, hidden_size=64, num_layers=1, output_len=pred_len)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 4. Predict
    with torch.no_grad():
        pred = model(input_tensor).squeeze(0).numpy()  # shape: (pred_len, 2)

    # 5. Return as DataFrame
    pred_df = pd.DataFrame(pred, columns=['x', 'y'])
    return pred_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict ping-pong ball trajectory using trained LSTM')
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV (columns: x, y)')
    parser.add_argument('--model', type=str, default='lstm_model.pth', help='Path to saved LSTM model file')
    parser.add_argument('--seq_len', type=int, default=10, help='Number of past points to use for prediction')
    parser.add_argument('--pred_len', type=int, default=5, help='Number of future points to predict')
    parser.add_argument('--output', type=str, default='predicted.csv', help='CSV file to save predicted points')
    args = parser.parse_args()

    # Run prediction
    predictions = predict_next(args.csv, args.model, args.seq_len, args.pred_len)
    predictions.to_csv(args.output, index=False)
    print(f"Predicted {args.pred_len} points saved to {args.output}")
