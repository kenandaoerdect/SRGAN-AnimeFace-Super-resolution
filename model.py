import torch
import torch.nn as nn
import torch.nn.functional as F


class Resblock(nn.Module):
    def __init__(self, channels):
        super(Resblock, self).__init__()
        self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.PReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels)
            )

    def forward(self, x):
        residual = self.residual(x)
        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(up_scale),
                nn.PReLU()
            )

    def forward(self, x):
        return self.upsample(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, padding=2),
                nn.PReLU()
            )
        self.resblocks = nn.Sequential(
                Resblock(64),
                Resblock(64),
                Resblock(64)
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.PReLU()
            )
        self.upsample = UpsampleBLock(64, 2)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=5, padding=2)


    def forward(self, x):
        block1 = self.conv1(x)
        block2 = self.resblocks(block1)
        block3 = self.conv2(block2)
        block4 = self.upsample(block1 + block3)
        block5 = self.conv3(block4)
        # return (torch.tanh(block5)+1) / 2
        return block5


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


if __name__ == '__main__':
    a = torch.randn(1, 3, 48, 48)
    net = Generator()
    net2 = Discriminator()
    out = net(a)
    print(out.shape)