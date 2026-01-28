from torch import nn

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.res = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.res(x) 
        return self.conv2(x)