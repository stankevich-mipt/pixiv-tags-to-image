import torch
import numpy as np
import os

from torch.utils.data import Dataset
from PIL import Image



class ImageBatch():

    def __init__(self, images): self.images = torch.stack(images, dim=0)
    
    def pin_memory(self): 
        
        self.images = self.images.pin_memory()
        return self

    def to(self, device):

        self.images = self.images.to(device)
        return self


class PixelCNNDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = os.path.join(os.getcwd(), root_dir)
        self.images   = sorted(os.listdir(self.root_dir)) 

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.root_dir, self.images[idx])
        img      = Image.open(img_name).resize((64, 64), Image.ANTIALIAS)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()