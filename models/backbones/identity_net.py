import torch.nn as nn


# 定义一个包含Identity层的网络
class IDentityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)  # 直接返回输入x
