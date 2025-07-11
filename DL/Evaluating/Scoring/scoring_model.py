import torch
import torch.nn as nn

class ScoringModel(nn.Module):
    """
    回归模型，用于根据融合特征预测击球评分。
    """
    def __init__(self, input_dim: int, hidden_dims=(128, 64)):
        """
        参数:
        - input_dim: 输入特征维度
        - hidden_dims: 隐藏层维度列表
        """
        super(ScoringModel, self).__init__()
        layers = []
        prev_dim = input_dim
        # 构建多层前馈网络：Linear -> GELU -> LayerNorm
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(h))
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        # 最终回归层，输出单值评分
        self.regressor = nn.Linear(prev_dim, 1)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算
        输入: x 形状 [B, input_dim]
        输出: y 形状 [B]，预测的评分
        """
        h = self.encoder(x)
        y = self.regressor(h)
        return y.view(-1)  # [B]

if __name__ == "__main__":
    # 快速测试
    dummy = torch.randn(8, 300)  # batch 8, 假设 input_dim=300
    model = ScoringModel(input_dim=300)
    out = model(dummy)
    print("Output shape:", out.shape)  # 应为 [8]
