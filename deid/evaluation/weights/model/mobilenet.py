"""MobileFaceNet backbone for eDifFIQA-T variant."""
import torch
from torch import nn


class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, k=(1, 1), s=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=(k[0] // 2, k[1] // 2), bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-05)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.prelu(x)


class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, k=(1, 1), s=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=(k[0] // 2, k[1] // 2), bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-05)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x)


class GDC_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c, eps=1e-05)
        self.prelu1 = nn.PReLU(in_c)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, groups=in_c, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c, eps=1e-05)
        self.prelu2 = nn.PReLU(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c, eps=1e-05)
        self.prelu3 = nn.PReLU(out_c)
        self.conv4 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(out_c, eps=1e-05)

    def forward(self, x):
        x1 = self.prelu1(self.bn1(self.conv1(x)))
        x1 = self.prelu2(self.bn2(self.conv2(x1)))
        x1 = self.prelu3(self.bn3(self.conv3(x1)))
        x2 = self.bn4(self.conv4(x))
        return torch.add(x1, x2)


class MobileFaceNet(nn.Module):
    def __init__(self, input_size=(112, 112), num_features=512):
        super().__init__()
        self.conv1 = Conv_block(3, 64, k=(3, 3), s=(2, 2))
        self.conv2_se = Conv_block(64, 64, k=(3, 3), s=(1, 1))
        self.conv3_dw = GDC_block(64, 64)
        self.conv4 = Conv_block(64, 64, k=(3, 3), s=(1, 1))
        self.conv5 = Conv_block(64, 64, k=(3, 3), s=(1, 1))
        self.conv6_dw = GDC_block(64, 128)
        self.conv7 = Conv_block(128, 128, k=(3, 3), s=(1, 1))
        self.conv8 = Conv_block(128, 128, k=(3, 3), s=(1, 1))
        self.conv9 = Conv_block(128, 128, k=(3, 3), s=(1, 1))
        self.conv10 = Conv_block(128, 128, k=(3, 3), s=(1, 1))
        self.conv11 = Conv_block(128, 128, k=(3, 3), s=(1, 1))
        self.conv12_dw = GDC_block(128, 256)
        self.conv13 = Conv_block(256, 256, k=(3, 3), s=(1, 1))
        self.conv14 = Linear_block(256, 512, k=(7, 7), s=(1, 1))
        self.embeddings = nn.BatchNorm1d(num_features, eps=1e-05)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """x: (B, 3, H, W). Returns (B, 512) embedding."""
        x = self.conv1(x)
        x = self.conv2_se(x)
        x = self.conv3_dw(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6_dw(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12_dw(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = x.view(x.size(0), -1)
        return self.embeddings(x)
