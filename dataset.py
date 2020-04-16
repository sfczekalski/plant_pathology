import numpy as np
import pandas as pd
import cv2
from tools import load_image

import torch
from torch.utils.data import Dataset, DataLoader


class PlantDataset(Dataset):
    def __init__(self, df, data_dir, transforms=None):
        self.df = df
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_path = self.data_dir + 'images/' + self.df.loc[idx, 'image_id'] + '.jpg'
        image = load_image(image_path=image_path)

        labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        labels = torch.from_numpy(labels.astype(np.int8))
        labels = labels.unsqueeze(-1)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

            return image, labels
