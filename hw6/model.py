import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmenterModel(nn.Module):
    def __init__(self):
        super(SegmenterModel, self).__init__()
        self.init_ch = 64 # число каналов после первой свёртки
        self.n_levels = 3 # число уровней до "основания" параболы

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module('conv1_input', nn.Conv2d(3, 64, 3, padding=1))
        self.layer_1.add_module('bn1_input', nn.BatchNorm2d(64))
        self.layer_1.add_module('relu1_in', nn.ReLU())
        self.layer_1.add_module('conv1', nn.Conv2d(64, 64, 3, padding=1))
        self.layer_1.add_module('bn1', nn.BatchNorm2d(64))
        self.layer_1.add_module('relu1', nn.ReLU())


        self.layer_2 = nn.Sequential()
        self.layer_2.add_module('pool2', nn.MaxPool2d(2))
        self.layer_2.add_module('conv2_input', nn.Conv2d(64, 128, 3, padding=1))
        self.layer_2.add_module('bn2_input', nn.BatchNorm2d(128))
        self.layer_2.add_module('relu2_in', nn.ReLU())
        self.layer_2.add_module('conv2', nn.Conv2d(128, 128, 3, padding=1))
        self.layer_2.add_module('bn2', nn.BatchNorm2d(128))
        self.layer_2.add_module('relu2', nn.ReLU())

        self.layer_3 = nn.Sequential()
        self.layer_3.add_module('pool3', nn.MaxPool2d(2))
        self.layer_3.add_module('conv3_input', nn.Conv2d(128, 256, 3, padding=1))
        self.layer_3.add_module('bn3_input', nn.BatchNorm2d(256))
        self.layer_3.add_module('relu3_in', nn.ReLU())
        self.layer_3.add_module('conv3', nn.Conv2d(256, 256, 3, padding=1))
        self.layer_3.add_module('bn3', nn.BatchNorm2d(256))
        self.layer_3.add_module('relu3', nn.ReLU())

        self.layer_4 = nn.Sequential()
        self.layer_4.add_module('pool4', nn.MaxPool2d(2))
        self.layer_4.add_module('conv4_input', nn.Conv2d(256, 512, 3, padding=1))
        self.layer_4.add_module('bn4_input', nn.BatchNorm2d(512))
        self.layer_4.add_module('relu4_in', nn.ReLU())
        self.layer_4.add_module('conv4', nn.Conv2d(512, 512, 3, padding=1))
        self.layer_4.add_module('bn4', nn.BatchNorm2d(512))
        self.layer_4.add_module('relu4', nn.ReLU())

        self.convtrans_1 = nn.ConvTranspose2d(512, 256, 2, 2)

        self.layer_5 = nn.Sequential()
        self.layer_5.add_module('conv5_input', nn.Conv2d(512, 256, 3, padding=1))
        self.layer_5.add_module('bn5_input', nn.BatchNorm2d(256))
        self.layer_5.add_module('relu5_in', nn.ReLU())
        self.layer_5.add_module('conv5', nn.Conv2d(256, 256, 3, padding=1))
        self.layer_5.add_module('bn5', nn.BatchNorm2d(256))
        self.layer_5.add_module('relu5', nn.ReLU())

        self.convtrans_2 = nn.ConvTranspose2d(256, 128, 2, 2)

        self.layer_6 = nn.Sequential()
        self.layer_6.add_module('conv6_input', nn.Conv2d(256, 128, 3, padding=1))
        self.layer_6.add_module('bn6_input', nn.BatchNorm2d(128))
        self.layer_6.add_module('relu6_in', nn.ReLU())
        self.layer_6.add_module('conv6', nn.Conv2d(128, 128, 3, padding=1))
        self.layer_6.add_module('bn6', nn.BatchNorm2d(128))
        self.layer_6.add_module('relu6', nn.ReLU())

        self.convtrans_3 = nn.ConvTranspose2d(128, 64, 2, 2)

        self.layer_7 = nn.Sequential()
        self.layer_7.add_module('conv7_input', nn.Conv2d(128, 64, 3, padding=1))
        self.layer_7.add_module('bn7_input', nn.BatchNorm2d(64))
        self.layer_7.add_module('relu7_in', nn.ReLU())
        self.layer_7.add_module('conv7', nn.Conv2d(64, 64, 3, padding=1))
        self.layer_7.add_module('bn7', nn.BatchNorm2d(64))
        self.layer_7.add_module('relu7', nn.ReLU())
        self.layer_7.add_module('conv7_out', nn.Conv2d(64, 1, 1))



    def forward(self, x):
        x1 = self.layer_1(x)
        x2 = self.layer_2(x1)
        x3 = self.layer_3(x2)
        x = self.layer_4(x3)
        x = self.convtrans_1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.layer_5(x)
        x = self.convtrans_2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.layer_6(x)
        x = self.convtrans_3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.layer_7(x)
        return x
    
    def predict(self, x):
        # на вход подаётся одна картинка, а не батч, поэтому так
        y = self.forward(x.unsqueeze(0).cuda())
        return (y > 0).squeeze(0).squeeze(0).float().cuda()
