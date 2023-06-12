import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

datatset = torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor(),
                                        download=True)
dataloader = DataLoader(datatset,batch_size=1)


class Detc(nn.Module):
    def __init__(self):
        super(Detc,self).__init__()
        self.module1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )


    def forward(self,x):
        x = self.module1(x)
        return x


loss = nn.CrossEntropyLoss()
XIAODAI = Detc()
optim = torch.optim.SGD(XIAODAI.parameters(),lr=0.01)
for epoch in range(20):
    runing_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = XIAODAI(imgs)
        result_loss = loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        runing_loss = runing_loss + result_loss
    print(runing_loss)







