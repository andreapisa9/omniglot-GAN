import torch
import torch.nn as nn


#MLP
class MLP(nn.Module):
    def __init__(self, nz = 100):
        super(MLP, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(nz, nz),
            nn.ReLU(),
            nn.Linear(nz, nz),
            nn.ReLU(),
            nn.Linear(nz, nz)
        )

    def forward(self, input):
        return self.main(input)

class MLP_mean_std(nn.Module):
    def __init__(self, in_channel = 256, nz = 100):
        super(MLP_mean_std, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_channel, nz),
            nn.ReLU(),
            nn.Linear(nz, nz),
            nn.ReLU()
        )

        self.mean = nn.Linear(nz, nz)
        self.std = nn.Linear(nz, nz)

    def forward(self, input):
        x = self.main(input)
        mean = self.mean(x)
        std = self.std(x)
        return mean, std

# MLP class
class MLP_cls(nn.Module):
    def __init__(self, in_channel = 256, nz = 100, out_channel = 30):
        super(MLP_cls, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_channel, nz),
            nn.ReLU(),
            nn.Linear(nz, nz),
            nn.ReLU(),
            nn.Linear(nz, out_channel)
        )

    def forward(self, input):
        return self.main(input)

# generator
class Generator(nn.Module):
    def __init__(self, nz = 100, ngf = 64, nc = 1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, input):
        return self.main(input)

# discriminator
class Discriminator(nn.Module):
    def __init__(self, nc = 1, ndf = 64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
        )

        self.conv1 = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        self.sigmoid1 = nn.Sigmoid()

        self.conv2 = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        self.sigmoid2 = nn.Sigmoid()

        self.avg_pool= nn.AvgPool2d(4)

    def forward(self, input):
        x = self.main(input)

        out_adv = self.conv1(x)
        out_adv = self.sigmoid1(out_adv)
        
        out_cls = self.conv2(x)
        out_cls = self.sigmoid2(out_cls)

        x = self.avg_pool(x)
        x = torch.mean(x, 0).squeeze()
        x = x.unsqueeze(0)
        x = x.repeat(input.size(0),1)

        return out_adv, out_cls, x

