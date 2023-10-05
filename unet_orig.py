
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import functional as F
import torch

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = torch.nn.Conv2d(64, 64, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')


        self.conv3 = torch.nn.Conv2d(64, 128, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        self.conv4 = torch.nn.Conv2d(128, 128, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')


        self.conv5 = torch.nn.Conv2d(128, 256, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv5.weight, nonlinearity='relu')
        self.conv6 = torch.nn.Conv2d(256, 256, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv6.weight, nonlinearity='relu')


        self.conv7 = torch.nn.Conv2d(256, 512, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv7.weight, nonlinearity='relu')
        self.conv8 = torch.nn.Conv2d(512, 512, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv8.weight, nonlinearity='relu')


        self.conv9 = torch.nn.Conv2d(512, 1024, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv9.weight, nonlinearity='relu')
        self.conv10 = torch.nn.Conv2d(1024, 1024, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv10.weight, nonlinearity='relu')

        self.dropout1 = torch.nn.Dropout(p = 0.1)
        self.dropout3 = torch.nn.Dropout(p = 0.3)

        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        x = F.relu(self.conv1(x))
        x =self.dropout1(x)
        x = F .relu(self.conv2(x))
        ftrs.append(x)
        x = self.pool(x)


        x = F.relu(self.conv3(x))
        x =self.dropout1(x)
        x = F.relu(self.conv4(x))
        ftrs.append(x)
        x = self.pool(x)


        x = F.relu(self.conv5(x))
        x =self.dropout1(x)
        x = F.relu(self.conv6(x))
        ftrs.append(x)
        x = self.pool(x)


        x = F.relu(self.conv7(x))
        x =self.dropout1(x)
        x = F.relu(self.conv8(x))
        ftrs.append(x)
        x = self.pool(x)


        x = F.relu(self.conv9(x))
        x =self.dropout3(x)
        x = F.relu(self.conv10(x))
        ftrs.append(x)
        return ftrs


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convTr1 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2,stride=2)
        self.conv1 = torch.nn.Conv2d(1024, 512, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.conv1_1 = torch.nn.Conv2d(512, 512, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1_1.weight, nonlinearity='relu')


        self.convTr2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2,stride=2)
        self.conv2 = torch.nn.Conv2d(512, 256, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.conv2_2 = torch.nn.Conv2d(256, 256, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2_2.weight, nonlinearity='relu')


        self.convTr3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2,stride=2)
        self.conv3 = torch.nn.Conv2d(256, 128, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        self.conv3_3 = torch.nn.Conv2d(128, 128, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv3_3.weight, nonlinearity='relu')


        self.convTr4 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2,stride=2)
        self.conv4 = torch.nn.Conv2d(128, 64, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')
        self.conv4_4 = torch.nn.Conv2d(64, 64, 3,padding=1)
        torch.nn.init.kaiming_uniform_(self.conv4_4.weight, nonlinearity='relu')

        self.dropout1 = torch.nn.Dropout(p = 0.1)
        self.dropout2 = torch.nn.Dropout(p = 0.2)




    def forward(self, x, encoder_features):
        x = self.convTr1(x)
        x = torch.cat([x, encoder_features[0]], dim=1)
        x = F.relu(self.conv1(x))
        x =self.dropout1(x)
        x = F.relu(self.conv1_1(x))


        x = self.convTr2(x)
        x = torch.cat([x, encoder_features[1]], dim=1)
        x = F.relu(self.conv2(x))
        x =self.dropout2(x)
        x = F.relu(self.conv2_2(x))


        x = self.convTr3(x)
        x = torch.cat([x, encoder_features[2]], dim=1)
        x = F.relu(self.conv3(x))
        x =self.dropout1(x)
        x = F.relu(self.conv3_3(x))


        x = self.convTr4(x)
        x = torch.cat([x, encoder_features[3]], dim=1)
        x = F.relu(self.conv4(x))
        x =self.dropout1(x)
        x = F.relu(self.conv4_4(x))
        return x


class UNet(torch.nn.Module):

    def __init__(self, num_class=1, retain_dim=False):
        super().__init__()
        self.encoder     = Encoder()
        self.decoder     = Decoder()
        self.head        = torch.nn.Conv2d(64, num_class, kernel_size=1)

        ##########################################
        # if you want to get the predicted probabilities of the classes, use softmax
        #self.softmax     = torch.nn.Softmax(dim=1)
        ###########################################

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        #out     = self.softmax(out)

        return out
