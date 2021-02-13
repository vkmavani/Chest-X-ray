# -*- coding: utf-8 -*-

'''   Import libraries   '''
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Create Model ---------- #

class CONV(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, dirate=1):
        super(CONV,self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners = True)
    return src

''' Network Components '''
### RSU-7 ###
class RSU7(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=1):
        super(RSU7, self).__init__()

        self.convin = CONV(in_ch, out_ch, dirate=1)

        self.conv1 = CONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv2 = CONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv3 = CONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv4 = CONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv5 = CONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv6 = CONV(mid_ch, mid_ch, dirate=1)

        self.conv7 = CONV(mid_ch, mid_ch, dirate=2)

        self.conv6d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv5d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv4d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv3d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv2d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv1d = CONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        xin = self.convin(x)

        x1 = self.conv1(xin)
        x = self.pool1(x1)

        x2 = self.conv2(x)
        x = self.pool2(x2)

        x3 = self.conv3(x)
        x = self.pool3(x3)

        x4 = self.conv4(x)
        x = self.pool4(x4)

        x5 = self.conv5(x)
        x = self.pool5(x5)

        x6 = self.conv6(x)

        x7 = self.conv7(x6)

        x6d =  self.conv6d(torch.cat((x7,x6), 1))
        x6dup = _upsample_like(x6d, x5)

        x5d =  self.conv5d(torch.cat((x6dup,x5), 1))
        x5dup = _upsample_like(x5d, x4)

        x4d = self.conv4d(torch.cat((x5dup,x4), 1))
        x4dup = _upsample_like(x4d, x3)

        x3d = self.conv3d(torch.cat((x4dup,x3), 1))
        x3dup = _upsample_like(x3d, x2)

        x2d = self.conv2d(torch.cat((x3dup,x2), 1))
        x2dup = _upsample_like(x2d, x1)

        x1d = self.conv1d(torch.cat((x2dup,x1), 1))
        return x1d + xin

### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=1):
        super(RSU6, self).__init__()

        self.convin = CONV(in_ch, out_ch, dirate=1)

        self.conv1 = CONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv2 = CONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv3 = CONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv4 = CONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv5 = CONV(mid_ch, mid_ch, dirate=1)

        self.conv6 = CONV(mid_ch, mid_ch, dirate=2)

        self.conv5d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv4d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv3d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv2d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv1d = CONV(mid_ch*2, out_ch, dirate=1)

    def forward(self,x):
        xin = self.convin(x)

        x1 = self.conv1(xin)
        x = self.pool1(x1)

        x2 = self.conv2(x)
        x = self.pool2(x2)

        x3 = self.conv3(x)
        x = self.pool3(x3)

        x4 = self.conv4(x)
        x = self.pool4(x4)

        x5 = self.conv5(x)

        x6 = self.conv6(x5)

        x5d =  self.conv5d(torch.cat((x6,x5), 1))
        x5dup = _upsample_like(x5d, x4)

        x4d = self.conv4d(torch.cat((x5dup,x4), 1))
        x4dup = _upsample_like(x4d, x3)

        x3d = self.conv3d(torch.cat((x4dup,x3), 1))
        x3dup = _upsample_like(x3d, x2)

        x2d = self.conv2d(torch.cat((x3dup,x2), 1))
        x2dup = _upsample_like(x2d, x1)

        x1d = self.conv1d(torch.cat((x2dup,x1), 1))
        return x1d + xin

### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=1):
        super(RSU5, self).__init__()

        self.convin = CONV(in_ch, out_ch, dirate=1)

        self.conv1 = CONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv2 = CONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv3 = CONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv4 = CONV(mid_ch, mid_ch, dirate=1)

        self.conv5 = CONV(mid_ch, mid_ch, dirate=2)

        self.conv4d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv3d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv2d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv1d = CONV(mid_ch*2, out_ch, dirate=1)

    def forward(self,x):
        xin = self.convin(x)

        x1 = self.conv1(xin)
        x = self.pool1(x1)

        x2 = self.conv2(x)
        x = self.pool2(x2)

        x3 = self.conv3(x)
        x = self.pool3(x3)

        x4 = self.conv4(x)

        x5 = self.conv5(x4)

        x4d = self.conv4d(torch.cat((x5,x4), 1))
        x4dup = _upsample_like(x4d, x3)

        x3d = self.conv3d(torch.cat((x4dup,x3), 1))
        x3dup = _upsample_like(x3d, x2)

        x2d = self.conv2d(torch.cat((x3dup,x2), 1))
        x2dup = _upsample_like(x2d, x1)

        x1d = self.conv1d(torch.cat((x2dup,x1), 1))
        return x1d + xin

### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=1):
        super(RSU4, self).__init__()

        self.convin = CONV(in_ch, out_ch, dirate=1)

        self.conv1 = CONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv2 = CONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.conv3 = CONV(mid_ch, mid_ch, dirate=1)

        self.conv4 = CONV(mid_ch, mid_ch, dirate=2)

        self.conv3d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv2d = CONV(mid_ch*2, mid_ch, dirate=1)
        self.conv1d = CONV(mid_ch*2, out_ch, dirate=1)

    def forward(self,x):
        xin = self.convin(x)

        x1 = self.conv1(xin)
        x = self.pool1(x1)

        x2 = self.conv2(x)
        x = self.pool2(x2)

        x3 = self.conv3(x)

        x4 = self.conv4(x3)

        x3d = self.conv3d(torch.cat((x4,x3), 1))
        x3dup = _upsample_like(x3d, x2)

        x2d = self.conv2d(torch.cat((x3dup,x2), 1))
        x2dup = _upsample_like(x2d, x1)

        x1d = self.conv1d(torch.cat((x2dup,x1), 1))
        return x1d + xin

### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=1, mid_ch=12, out_ch=1):
        super(RSU4F, self).__init__()

        self.convin = CONV(in_ch, out_ch, dirate=1)

        self.conv1 = CONV(out_ch, mid_ch, dirate=1)
        self.conv2 = CONV(mid_ch, mid_ch, dirate=2)
        self.conv3 = CONV(mid_ch, mid_ch, dirate=4)

        self.conv4 = CONV(mid_ch, mid_ch, dirate=8)

        self.conv3d = CONV(mid_ch*2, mid_ch, dirate=4)
        self.conv2d = CONV(mid_ch*2, mid_ch, dirate=2)
        self.conv1d = CONV(mid_ch*2, out_ch, dirate=1)

    def forward(self,x):
        xin = self.convin(x)

        x1 = self.conv1(xin)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x4 = self.conv4(x3)

        x3d = self.conv3d(torch.cat((x4,x3), 1))
        x2d = self.conv2d(torch.cat((x3d,x2), 1))
        x1d = self.conv1d(torch.cat((x2d,x1), 1))
        return x1d + xin


#### U2-Net ###
class U2NET_L(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super(U2NET_L, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool1_2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool2_3 = nn.MaxPool3d(2, stride=2,ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool3_4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool4_5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool5_6 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv3d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv3d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv3d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv3d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv3d(6, out_ch, 1)

    def forward(self, x):
        #stage 1
        x1 = self.stage1(x)
        x = self.pool1_2(x1)

        #stage 2
        x2 = self.stage2(x)
        x = self.pool2_3(x2)

        #stage 3
        x3 = self.stage3(x)
        x = self.pool3_4(x3)

        #stage 4
        x4 = self.stage4(x)
        x = self.pool4_5(x4)

        #stage 5
        x5 = self.stage5(x)
        x = self.pool5_6(x5)

        #stage 6
        x6 = self.stage6(x)
        x6up = _upsample_like(x6, x5)

        # ------------------- decoder ------------------- #
        x5d = self.stage5d(torch.cat((x6up,x5), 1))
        x5dup = _upsample_like(x5d, x4)

        x4d = self.stage4d(torch.cat((x5dup,x4), 1))
        x4dup = _upsample_like(x4d, x3)

        x3d = self.stage3d(torch.cat((x4dup,x3), 1))
        x3dup = _upsample_like(x3d, x2)

        x2d = self.stage2d(torch.cat((x3dup,x2), 1))
        x2dup = _upsample_like(x2d, x1)

        x1d = self.stage1d(torch.cat((x2dup,x1), 1))

        # ----------------- side output ----------------- #
        d1 = self.side1(x1d)

        d2 = self.side2(x2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(x3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(x4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(x5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(x6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6), 1))
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

### U2-Net small ###
class U2NET(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super(U2NET, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool1_2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool2_3 = nn.MaxPool3d(2, stride=2,ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool3_4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool4_5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool5_6 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv3d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv3d(6, out_ch, 1)


    def forward(self, x):
        #stage 1
        x1 = self.stage1(x)
        x = self.pool1_2(x1)

        #stage 2
        x2 = self.stage2(x)
        x = self.pool2_3(x2)

        #stage 3
        x3 = self.stage3(x)
        x = self.pool3_4(x3)

        #stage 4
        x4 = self.stage4(x)
        x = self.pool4_5(x4)

        #stage 5
        x5 = self.stage5(x)
        x = self.pool5_6(x5)

        #stage 6
        x6 = self.stage6(x)
        x6up = _upsample_like(x6, x5)

        # ------------------- decoder ------------------- #
        x5d = self.stage5d(torch.cat((x6up,x5), 1))
        x5dup = _upsample_like(x5d, x4)

        x4d = self.stage4d(torch.cat((x5dup,x4), 1))
        x4dup = _upsample_like(x4d, x3)

        x3d = self.stage3d(torch.cat((x4dup,x3), 1))
        x3dup = _upsample_like(x3d, x2)

        x2d = self.stage2d(torch.cat((x3dup,x2), 1))
        x2dup = _upsample_like(x2d, x1)

        x1d = self.stage1d(torch.cat((x2dup,x1), 1))

        # ----------------- side output ----------------- #
        d1 = self.side1(x1d)

        d2 = self.side2(x2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(x3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(x4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(x5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(x6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6), 1))

        if self.training:
            return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)
        else:
            return torch.sigmoid(d0)
