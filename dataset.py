# dataset.py
import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import h5py
import torchvision.transforms as transforms

class listDataset(Dataset):
    def __init__(self, root, shuffle=False, transform=None, train=False, seen=0, batch_size=1, num_workers=4):
        self.root = root
        self.transform = transform
        self.train = train
        self.batch_size = batch_size
        self.seen = seen
        self.num_workers = num_workers
        self.images = glob.glob(os.path.join(root, '*.jpg')) + glob.glob(os.path.join(root, '*.png'))
        if shuffle:
            np.random.shuffle(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')

        gt_path = img_path.replace('.jpg', '.h5').replace('.png', '.h5')
        with h5py.File(gt_path, 'r') as gt_file:
            target = np.asarray(gt_file['density'])

        if self.transform is not None:
            img = self.transform(img)

        target = torch.from_numpy(target).unsqueeze(0).float()

        return img, target

    def __len__(self):
        return len(self.images)
