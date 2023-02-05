import torch
import numpy as np
import pandas as pd
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        df = pd.read_csv(root, encoding='gbk', header=None)
        self.imgs1 = df[0].tolist()
        self.imgs2 = df[1].tolist()
        self.targets = df[2].tolist()
        self.classes = ['False', 'True']
        self.class_to_idx = {'False': 0, 'True': 1}
        self.transform = transform
    
    def __getitem__(self, idx):
        target = np.array(self.targets[idx], dtype=np.float32)
        img1 = Image.open(self.imgs1[idx]).convert('L')
        img2 = Image.open(self.imgs2[idx]).convert('L')
        if self.transform is not None:
            img1 = 1 - self.transform(img1)
            img2 = 1 - self.transform(img2)
        return img1, img2, target
    
    def __len__(self):
        return len(self.imgs1)