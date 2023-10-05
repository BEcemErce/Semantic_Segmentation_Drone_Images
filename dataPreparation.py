import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms


class dataPrep(Dataset):
    def __init__(self, imagePaths,maskPaths,n_class,W,H):
        self.imagePaths = imagePaths
        self.maskPaths= maskPaths
        self.W=W
        self.H=H
        self.n_class=n_class

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]

        #transform the original image
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        if image.size is None:
            return None
        image=image/255.0
        image=cv2.resize(image,(self.W,self.H))
        image = image.astype(np.float32)

        #transform the masked image
        mask = cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE)
        if mask.size is None:
            return None
        mask=cv2.resize(mask,(self.W,self.H))
        mask = mask.astype(np.int32)

        #transform the original and masked images to tensor
        tensor_transform=transforms.ToTensor()
        image=tensor_transform(image)
        mask=tensor_transform(mask)
        mask=torch.squeeze(mask)
        mask_orig=mask.long()

        ########################################################################################################
        # if you need the one hot encodes for the masks, use this code lines and put the result to return part.#
        #mask=torch.nn.functional.one_hot(mask.to(torch.int64), num_classes=self.n_class)
        #mask=mask.view(self.n_class,self.H,self.W).float()
        ########################################################################################################

        return (image, mask_orig)