import torch
from torch import nn

class MainNet(torch.nn.Module):

    def __init__(self,):
        super(MainNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 256),

            nn.LayerNorm(256),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(256, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, 7),
        )

    def forward(self, x):


        out = self.fc(x)
        return out


if __name__ == '__main__':
    import time

    x = torch.rand(3, 7).cuda()

    net = MainNet().cuda()
    print(net(x).shape)
