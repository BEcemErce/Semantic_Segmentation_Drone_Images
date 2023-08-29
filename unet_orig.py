# -*- coding: utf-8 -*-
"""
"""
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch,torchvision

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

        self.conv1 = torch.nn.Conv2d(3, 16, 3,padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3,padding=1)
        self.bn1=torch.nn.BatchNorm2d(16)

        self.conv3 = torch.nn.Conv2d(16, 32, 3,padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3,padding=1)
        self.bn2=torch.nn.BatchNorm2d(32)

        self.conv5 = torch.nn.Conv2d(32, 64, 3,padding=1)
        self.conv6 = torch.nn.Conv2d(64, 64, 3,padding=1)
        self.bn3=torch.nn.BatchNorm2d(64)
   
        
        self.conv7 = torch.nn.Conv2d(64, 128, 3,padding=1)
        self.conv8 = torch.nn.Conv2d(128, 128, 3,padding=1)
        self.bn4=torch.nn.BatchNorm2d(128)
        
        
        self.conv9 = torch.nn.Conv2d(128, 256, 3,padding=1)
        self.conv10 = torch.nn.Conv2d(256, 256, 3,padding=1)
        self.bn5=torch.nn.BatchNorm2d(256)

        #self.conv7 = torch.nn.Conv2d(256, 512, 3,padding=1)
        #self.conv8 = torch.nn.Conv2d(512, 512, 3,padding=1)

        self.dropout1 = torch.nn.Dropout(p = 0.1) 
        self.dropout3 = torch.nn.Dropout(p = 0.3) 

        self.pool = torch.nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        x = self.conv1(x)
        x = torch.nn.functional .relu(x)
        x =self.dropout1(x)
        x = self.conv2(x)
        x = torch.nn.functional .relu(x)
        ftrs.append(x)
        x = self.pool(x)
        
        
        x = self.conv3(x)
        x = torch.nn.functional .relu(x)
        x =self.dropout1(x)
        x = self.conv4(x)
        x = torch.nn.functional .relu(x)
        ftrs.append(x)
        x = self.pool(x)
        
        
        x = self.conv5(x)
        x = torch.nn.functional .relu(x)
        x =self.dropout1(x)
        x = self.conv6(x)
        x = torch.nn.functional .relu(x)
        ftrs.append(x)
        x = self.pool(x)
        

        x = self.conv7(x)
        x = torch.nn.functional .relu(x)
        x =self.dropout1(x)
        x = self.conv8(x)
        x = torch.nn.functional .relu(x)
        ftrs.append(x)
        x = self.pool(x)
        

        x = self.conv9(x)
        x = torch.nn.functional .relu(x)
        x =self.dropout3(x)
        x = self.conv10(x)
        x = torch.nn.functional .relu(x)
        ftrs.append(x)
        #x = self.pool(x)
        
    
        return ftrs


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #self.convTr0 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2,stride=2)
        #self.conv0= torch.nn.Conv2d(512, 256, 3,padding=1)
        
        self.convTr1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2,stride=2)
        self.conv1 = torch.nn.Conv2d(256, 128, 3,padding=1)
        self.conv1_1 = torch.nn.Conv2d(128, 128, 3,padding=1)
        self.bn1=torch.nn.BatchNorm2d(128)

        
        self.convTr2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2,stride=2)
        self.conv2 = torch.nn.Conv2d(128, 64, 3,padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, 3,padding=1)
        self.bn2=torch.nn.BatchNorm2d(64)

        self.convTr3 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2,stride=2)
        self.conv3 = torch.nn.Conv2d(64, 32, 3,padding=1)
        self.conv3_3 = torch.nn.Conv2d(32, 32, 3,padding=1)
        self.bn3=torch.nn.BatchNorm2d(32)

        self.convTr4 = torch.nn.ConvTranspose2d(32, 16, kernel_size=2,stride=2)
        self.conv4 = torch.nn.Conv2d(32, 16, 3,padding=1)
        self.conv4_4 = torch.nn.Conv2d(16, 16, 3,padding=1)
        self.bn4=torch.nn.BatchNorm2d(16)

        self.dropout1 = torch.nn.Dropout(p = 0.1) 
        self.dropout2 = torch.nn.Dropout(p = 0.2) 

     
        
    def forward(self, x, encoder_features):
        

        x = self.convTr1(x)
        #enc_ftrs = self.crop(encoder_features[0], x)  
        x        = torch.cat([x, encoder_features[0]], dim=1)
        x        = self.conv1(x)
        x = torch.nn.functional .relu(x)
        x =self.dropout1(x)
        x        = self.conv1_1(x)
        x = torch.nn.functional .relu(x)


        x = self.convTr2(x)
        #enc_ftrs = self.crop(encoder_features[1], x)  
        x        = torch.cat([x, encoder_features[1]], dim=1)
        x        = self.conv2(x)
        x = torch.nn.functional .relu(x)
        x =self.dropout2(x)
        x        = self.conv2_2(x)
        x = torch.nn.functional .relu(x)


        x = self.convTr3(x)
        #enc_ftrs = self.crop(encoder_features[2], x)  
        x        = torch.cat([x, encoder_features[2]], dim=1)
        x        = self.conv3(x)
        x = torch.nn.functional .relu(x)
        x =self.dropout1(x)
        x        = self.conv3_3(x)
        x = torch.nn.functional .relu(x)


        x = self.convTr4(x)
        x        = torch.cat([x, encoder_features[3]], dim=1)
        x        = self.conv4(x)
        x = torch.nn.functional .relu(x)
        x =self.dropout1(x)
        x        = self.conv4_4(x)
        x = torch.nn.functional .relu(x)

        return x
    
        
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(torch.nn.Module):
  
    def __init__(self, num_class=1, retain_dim=False):
        super().__init__()
        self.encoder     = Encoder()
        self.decoder     = Decoder()
        self.head        = torch.nn.Conv2d(16, num_class, kernel_size=1)
        self.softmax     = torch.nn.Softmax(dim=1)
        #self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        out     = self.softmax(out)
        
        #if self.retain_dim:
        #    out = torch.nn.functional.interpolate(out, (800,1200))
        return out
