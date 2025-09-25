import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class POOL(nn.Module):
    def __init__(self, embedding_dim=256, level=[1, 3, 5]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.level = level
        self.fc = nn.Sequential(L2Norm(dim=-1),
                                nn.Linear(embedding_dim * len(level),
                                          embedding_dim, bias=True),
                                L2Norm(dim=-1))
        self.attention_pool = nn.Linear(embedding_dim * len(level), len(level))

    def forward(self, x):  # (B, L_all, 1+14*14, C)

        x = x[:, self.level, :, :]  # (B, L, 1+14*14, C)
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), x.size(2), -1)  # (B,1+14*14,C*L)
        x, mask = self.pool1d(x)  # (B,C*L)
        x = self.fc(x)
        # 修改后的版本并不返回mask
        return x

    def pool1d(self, x):  # (B, 1+14*14, C) => (B,C)

        x = x[..., 1:, :]  # (B,14*14,C) or (B,14*14,C*L)

        mask = F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2)  # (B, 1, 14*14)/(B, L, 14*14)
        features = []
        for i in range(len(self.level)):
            feature = torch.matmul(mask[:, i:i + 1, :],
                                   x[..., i * self.embedding_dim:(i + 1) * self.embedding_dim]) \
                .squeeze(-2)
            # (B,1,14*14) mul (B,14*14,C) = (B,1,C) => (B,C)
            features.append(feature)
        x = torch.cat(features, -1)  # (B,C*L)
        return x, mask
