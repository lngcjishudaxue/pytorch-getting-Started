import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *


device = torch.device("cuda")

train_data = torchvision.datasets.CIFAR10("data_train",train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.CIFAR10("data_train",train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,64)

xiaodai1 = dai()
xiaodai1.to(device)


loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#优化器
learning_rate =0.01
optimizer = torch.optim.SGD(xiaodai1.parameters(),learning_rate)

#设置参数
total_train_step = 0
total_test_step = 0
epoch = 50

writer = SummaryWriter("logs")

for i in range(epoch):
    print("------第{}轮训练开始------".format(i+1))
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = xiaodai1(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    total_test_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = xiaodai1(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()
            acc = (outputs.argmax(1) == targets).sum()
            total_acc = total_acc + acc
    print("整体测试集上的正确率：{}".format(total_acc/test_data_size))
    print("测试数据集上的loss：{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step = total_test_step + 1
    is_best = 0
    if is_best < total_acc/test_data_size:
        is_best = total_acc/test_data_size
        print(is_best)
        torch.save(xiaodai1,"best.pth")

torch.save(xiaodai1,"last.pth")
writer.close()




#model = torchvision.models.vgg16(pre)

