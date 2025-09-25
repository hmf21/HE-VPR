import torchvision.transforms as T
import torch
import random
import numpy as np
from PIL import Image
import random


# 文章工作的内容之一，添加随机掩码
class RandomMaskPatches:
    # 不再显式继承Transform
    def __init__(self, mask_ratio=0.75, patch_size=32):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def __call__(self, img: Image.Image) -> Image.Image:  # 显式声明输入输出类型
        if self.mask_ratio == 0:
            return img
        h, w = img.size
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        total_patches = num_patches_h * num_patches_w

        num_mask_patches = int(total_patches * self.mask_ratio)
        mask_indices = random.sample(range(total_patches), k=num_mask_patches)

        # 创建掩码矩阵
        mask = torch.zeros((num_patches_h, num_patches_w), dtype=torch.bool)
        for idx in mask_indices:
            i = idx // num_patches_w
            j = idx % num_patches_w
            mask[i, j] = True

        # 上采样掩码到图像尺寸
        mask = mask.repeat_interleave(self.patch_size, dim=0).repeat_interleave(self.patch_size, dim=1)

        # 应用掩码，这里会要求掩码的大小与像素分辨率整除，否则会报错
        masked_img = img.copy()  # 使用PIL的copy方法
        masked_img = masked_img.convert("RGB")  # 确保图像模式正确
        img_array = torch.from_numpy(np.array(masked_img)).permute(2, 0, 1)  # HWC -> CHW
        expanded_mask = mask.expand(3, -1, -1)  # [C, H, W]
        img_array[expanded_mask] = 0  # 应用掩码
        masked_img = Image.fromarray(img_array.permute(1, 2, 0).numpy().astype(np.uint8))  # 转回PIL格式

        return masked_img

if __name__ == '__main__':
    img = Image.open("../asset/test.png").convert("RGB")
    img_rz = img.resize((512, 512))
    transform = RandomMaskPatches(mask_ratio=0.25)
    masked_img = transform(img_rz)
    masked_img.show()