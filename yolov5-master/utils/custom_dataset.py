import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPYDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy') or f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        
        if file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
        elif file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=True)
            data = {key: data[key] for key in data}
        
        image = data['image']
        labels = data['labels']

        if self.transform:
            transformed = self.transform(image=image, bboxes=labels)
            image = transformed['image']
            labels = transformed['bboxes']

        return image, labels
