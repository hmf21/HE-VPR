import torch
import torch.nn.functional as F
import torch.nn as nn

class GeMPoolDino(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.fc = nn.Linear(768, 128)


    def forward(self, x):
        t = x["x_norm_clstoken"]
        x = x["x_norm_patchtokens"].reshape(((t.shape)[0], 16, 16, (t.shape)[-1])).permute(0, 3, 1, 2)
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        x = x.flatten(1)
        x = x + t

        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x