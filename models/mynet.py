import torchvision
import torch
from torch import nn
import numpy as np

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size,
                      kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class mynet(nn.Module):
    def __init__(self, num_classes):
        super(mynet, self).__init__()
        pretrained_net = torchvision.models.resnet34(pretrained=True)
        self.stage1 = list(pretrained_net.children())[:-5]  # 第一段
        self.stage1[0] = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.stage1[-2] = nn.Conv2d(64, 64, 2, stride=2, padding=1, bias=False, dilation=2)
        temp = self.stage1[-1]
        self.stage1[-1] = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.stage1.append(temp)
        self.stage1 = nn.Sequential(*self.stage1)
        self.stage2 = list(pretrained_net.children())[-5]  # 第二段
        self.stage3 = list(pretrained_net.children())[-4]  # 第三段
        self.stage4 = list(pretrained_net.children())[-3]  # 第四段

        #改变通道数
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)
        self.scores4 = nn.Conv2d(64, num_classes, 1)
        
        #8倍上采样
        self.upsample_8x = nn.ConvTranspose2d(
            num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(
            num_classes, num_classes, 16)  
        
        #2倍上采样
        self.upsample_2x = nn.ConvTranspose2d(
            num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(
            num_classes, num_classes, 4)  

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 1/4   64
        x = self.stage2(x)
        s2 = x  # 1/8   128
        x = self.stage3(x)
        s3 = x  # 1/16  256
        x = self.stage4(x)
        s4 = x  # 1/32   512
        
        s4 = self.scores1(s4)
        s4 = self.upsample_2x(s4)  #1/16
        s3 = self.scores2(s3)
        s3 = s3 + s4
        s3 = self.upsample_2x(s3)  #1/8
        s2 = self.scores3(s2)
        s2 = s3 + s2
        s2 = self.upsample_2x(s2)  #1/4
        s1 = self.scores4(s1)
        s1 = s2 + s1
        s = self.upsample_2x(s1) 
        s = self.upsample_2x(s) 
        return s