import torch
from torch import nn


class dai(nn.Module):
    def __init__(self):
        super(dai,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    xiaodai = dai()
    input = torch.ones((64,3,32,32))
    output = xiaodai(input)
    print(output.shape)