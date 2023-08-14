import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import cv2
import numpy as np

class dataPrep(Dataset):
    def __init__(self, imagePaths,maskPaths,n_class, input_transform=None, mask_transform=None):
        self.imagePaths = imagePaths
        self.maskPaths= maskPaths
        self.input_transform = input_transform
        self.mask_transform = mask_transform
        self.n_class=n_class
        
    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
		# grab the image path from the current index
        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(maskPath)
        # check to see if we are applying any transformations
        if self.input_transform is not None:
            # apply the transformations to both image and its mask
            image = self.input_transform(image)


        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        # return a tuple of the image and its mask
        mask,_ =torch.max(mask,dim=0)
       
        h, w = mask.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
           target[c][mask == c] = 1
        

        return (image, target, mask)