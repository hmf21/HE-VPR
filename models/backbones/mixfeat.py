import torch
import torch.nn as nn
import numpy as np
import math
from models.backbones.dinov2 import DINOv2
from models.backbones.resnet import ResNet


class MixFeat(nn.Module):
    def __init__(self, backbone_arch='mixfeat'):
        super().__init__()

        # 都采用默认的配置'vit-s+resnet50'，为了尽量保证通道数的一致，选用Resnet50并抛弃最后两层
        self.VitPart = DINOv2(model_name='dinov2_vits14', norm_layer=True, return_token=False)
        self.CNNPart = ResNet(model_name='resnet50', layers_to_crop=[3, 4])

        self.mixfeat_res_wh = 16
        self.VitResAdapLayer = nn.AdaptiveAvgPool2d((self.mixfeat_res_wh, self.mixfeat_res_wh)) # feat_dim 512
        self.CNNResAdapLayer = nn.AdaptiveAvgPool2d((self.mixfeat_res_wh, self.mixfeat_res_wh)) # feat_dim 384

        self.out_channels = 512
        self.cnn_feat_dim = 512
        self.vit_feat_dim = 384

        self.CNNDimAlignConv = nn.Conv2d(self.cnn_feat_dim, self.out_channels, 1)
        self.VitDimAlignConv = nn.Conv2d(self.vit_feat_dim, self.out_channels, 1)

        # self.FeatMixerLayer = CrossAttention(dim=self.out_channels)
        self.FeatMixerLayer = nn.MultiheadAttention(self.out_channels, 4)

    def forward(self, x):
        cnn_feat = self.CNNPart(x)
        cnn_feat_adap_ = self.CNNResAdapLayer(cnn_feat)
        cnn_feat_adap = self.CNNDimAlignConv(cnn_feat_adap_)
        vit_feat = self.VitPart(x)
        vit_feat_adap_ = self.VitResAdapLayer(vit_feat)
        vit_feat_adap = self.VitDimAlignConv(vit_feat_adap_)

        output_feat = torch.cat((cnn_feat_adap, vit_feat_adap), dim=1)
        return output_feat


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 3, 224, 224).cuda()
    m = MixFeat().cuda()
    r = m(x)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')


if __name__ == '__main__':
    main()


