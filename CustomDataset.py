from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torch

class CustomDataset(Dataset):
    """Custom Dataset for loading IMDB-WIKI face images"""

    def __init__(self, csv_path, img_dir, NUM_AGE_CLASSES, MIN_AGE, transform=None):

        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['img_paths'].values
        self.ages = df['ages'].values
        self.genders = df['genders'].values
        self.transform = transform
        self.NUM_AGE_CLASSES = NUM_AGE_CLASSES
        self.MIN_AGE = MIN_AGE

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        age_label = self.ages[index]
        age_class = age_label - self.MIN_AGE
        # Map age to one of the predefined calsses
        # TODO: make this part more automated and cleaner
        # if age <= 24:
        #     age_label = 0
        # elif age <= 34:
        #     age_label = 1
        # elif age <= 49:
        #     age_label = 2
        # else:
        #     age_label = 3
        age_levels = [1]*age_class + [0]*(self.NUM_AGE_CLASSES - 1 - age_class)
        age_levels = torch.tensor(age_levels, dtype=torch.float32)

        gender_label = self.genders[index]

        return img, age_label, age_levels, gender_label

    def __len__(self):
        return self.ages.shape[0]