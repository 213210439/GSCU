import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.nn.functional as F
from gscu import GaussianSplatter

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, kernel_num):
        super(ResidualBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=in_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.Conv2 = nn.Conv2d(in_channels=in_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)

    def forward(self, x):
        y = F.relu(self.Conv1(x), False)
        y = self.Conv2(y)
        return x + y

class PanNet_GSCU(nn.Module):

    def __init__(self, channel=4,kernel=5,num_points=81,c1=8,n_feats=48,kernel_size=(3,3), kernel_num=32):
        super(PanNet_GSCU, self).__init__()
        self.GaussianSR =GaussianSplatter(kernel_size=kernel,num_points=num_points,c1=c1,channels=channel,n_feats=n_feats)
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)

    def forward(self, pan, ms, hpan):
        up_ms = self.GaussianSR(ms, pan)
        x = torch.cat([hpan, up_ms], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + up_ms, up_ms

class PanNet(nn.Module):

    def __init__(self, channel=4, kernel_size=(3,3), kernel_num=32):
        super(PanNet, self).__init__()
        # Conv2d默认stride=1, bias=True
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)


    def forward(self, pan, ms, hms, hpan):
        x_ms = fun.interpolate(ms, scale_factor=(4,4), mode='bicubic')
        up_ms = self.ConvTrans(hms)
        x = torch.cat([hpan, up_ms], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + x_ms

class PanNet_nearest(nn.Module):

    def __init__(self, channel=4, kernel_size=(3,3), kernel_num=32):
        super(PanNet_nearest, self).__init__()
        # Conv2d默认stride=1, bias=True
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)


    def forward(self, pan, ms, hms, hpan):
        x_ms = fun.interpolate(ms, scale_factor=(4,4), mode='nearest')
        up_ms = self.ConvTrans(hms)
        x = torch.cat([hpan, up_ms], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + x_ms

class PanNet_Tconv(nn.Module):

    def __init__(self, channel=4, kernel_size=(3,3), kernel_num=32):
        super(PanNet_Tconv, self).__init__()
        # Conv2d默认stride=1, bias=True
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)


    def forward(self, pan, ms, hms, hpan):
        x_ms = self.ConvTrans(ms)
        up_ms = self.ConvTrans(hms)
        x = torch.cat([hpan, up_ms], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + x_ms

if __name__ == '__main__':
    ms = torch.rand(1, 4, 32, 32)
    hms = torch.rand(1,4,32,32)
    pan = torch.rand(1, 1, 128, 128)
    hpan = torch.rand(1,1,128,128)
    PanNet_G = PanNet_GSCU()
    pred, x_ms = PanNet_G.forward(pan, ms, hpan)
    print("PanNet_GSCU输出维度为：", pred.shape)

    PanNet = PanNet(channel=4)
    pred = PanNet.forward(pan, ms, hms, hpan)
    print("正常PanNet的输出维度为：", pred.shape)

    PanNet_n = PanNet_nearest()
    pred = PanNet_n.forward(pan, ms, hms, hpan)
    print("PanNet_Nearest的输出维度为：", pred.shape)

    PanNet_T = PanNet_Tconv()
    pred = PanNet_T.forward(pan, ms, hms, hpan)
    print("PanNet_Tconv的输出维度为：", pred.shape)
