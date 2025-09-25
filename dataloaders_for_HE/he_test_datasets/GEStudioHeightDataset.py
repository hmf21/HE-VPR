from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat
import numpy as np
import torchvision.transforms as T
import torch.utils.data as data
from glob import glob
import re
import os
import torchvision.transforms as tvf
from PIL import Image
from sklearn.neighbors import NearestNeighbors

dataset_dir = './GEStudioHeightDataset'



class GEStuidioHEDataset(data.Dataset):
    def __init__(self, img_rz=(256, 256)):
        super().__init__()

        self.input_transform = T.Compose([
            T.Resize(img_rz, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.test_db_dir = join(dataset_dir, 'height_database_large')
        self.test_qr_dir = join(dataset_dir, 'query_images')

        self.db_paths = glob(join(self.test_db_dir, "*.tif"), recursive=True)
        self.qr_paths = glob(join(self.test_qr_dir, "*.png"), recursive=True)
        self.images = self.db_paths + self.qr_paths

        self.db_utms = np.array([(re.findall("\d+", path)[-1], 0) for path in self.db_paths]).astype(float)
        self.qr_utms = np.array([(float(path.split("@")[4])*0.357+5.33 , 0) for path in self.qr_paths]).astype(float)

        self.numDb = len(self.db_utms)
        self.numQ = len(self.qr_utms)

        self.positives = None
        self.distances = None

        self.dataset_info = {
            'num_db': self.numDb,
            'num_qr': self.numQ,
            'dataset_name': 'GStuidioHeightDataset',
            'dataset_dir': dataset_dir
        }

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if index < self.numDb:
            pass
        else:
            # 切成正方形
            img = img.crop(((img.size[0] - img.size[1]) / 2, 0, (img.size[0] + img.size[1]) / 2, img.size[1]))
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_utms)
            query_positive = 100
            self.distances, self.positives = knn.radius_neighbors(self.qr_utms, radius=query_positive)
        return self.positives


if __name__ == '__main__':
    test_dataset = GEStuidioHeightDataset()
    data_sample = test_dataset.__getitem__(1000)
    data_positive = test_dataset.getPositives()
    print(test_dataset.dataset_info)
    print(data_sample, data_positive[88].shape)
