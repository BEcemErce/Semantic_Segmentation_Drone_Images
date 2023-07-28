import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__ (self, n_class):
        super().__init__()


        ### ENCODER ###

        #Actually, this is VGG-16. VGG-16 have 16 layers but, I used only convolution part.

        self.conv1= nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2= nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1= nn.MaxPool2d(2,2)
    
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
    

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)


        ### DECODER ###

        
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        


        # this is the main difference of FCN from fully connected layers. In FCN, 1X1 kernels are used.
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)


    def forward(self,x):

        ### ENCODER ###

        x = F.relu(self.conv1(x))
        print("conv1",x.shape)
        x = F.relu(self.conv2(x))
        print("conv2",x.shape)
        x = self.pool1(x)
        print("pool1_out",x.shape)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        print("pool2_out",x.shape)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool3(x)
        print("pool3_out",x.shape)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        pool4_out = self.pool4(x)
        print("pool4_out",pool4_out.shape)

        x = F.relu(self.conv11(pool4_out))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        print("conv13",x.shape)
        pool5_out = self.pool5(x)

        print("pool5_out",pool5_out.shape)


        score = self.bn1(F.relu(self.deconv1(pool5_out)))     
        print("bn1",score.shape)
        score = self.bn2(F.relu(self.deconv2(score)))  
        print("bn2",score.shape)

        score = self.bn3(F.relu(self.deconv3(score)))
        print("bn3",score.shape)

        score = self.bn4(F.relu(self.deconv4(score)))  
        print("bn4",score.shape)
        score = self.bn5(F.relu(self.deconv5(score)))  
        print("bn5",score.shape)
        
        score = self.classifier(score)                    

        return score 

