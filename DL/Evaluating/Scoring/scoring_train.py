import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader_scoring import get_data_loaders
from scoring_model import ScoringModel


def train(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        root_dir=args.data_root,
        window_size=args.window_size,
        batch_size=args.batch_size,
        splits=(args.train_split, args.val_split, 1 - args.train_split - args.val_split),
        num_workers=args.num_workers
    )

    # 实例化模型
    # 从任一 batch 中获取特征维度
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[1]
    model = ScoringModel(input_dim=input_dim, hidden_dims=args.hidden_dims).to(device)

    # 损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # 训练模式
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # 验证模式
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds_val = model(X_val)
                loss_val = criterion(preds_val, y_val)
                val_losses.append(loss_val.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, 'best_scoring_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    print("Training completed.")

    # 测试集评估
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_scoring_model.pth')))
    model.eval()
    test_losses = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            preds_test = model(X_test)
            loss_test = criterion(preds_test, y_test)
            test_losses.append(loss_test.item())
    print(f"Test Loss: {sum(test_losses) / len(test_losses):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train scoring model')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of aligned data')
    parser.add_argument('--window_size', type=int, default=50, help='Fixed window size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--train_split', type=float, default=0.6, help='Train split fraction')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split fraction')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='results/models', help='Directory to save checkpoints')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer dimensions')
    args = parser.parse_args()
    train(args)
