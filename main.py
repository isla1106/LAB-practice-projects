import torch
import torchvision
import torchvision.transforms as transforms
import ssl
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

#参数
#cuda可行情况下使用gpu训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 选定要使用的模型
batch_size = 64
learning_rate = 0.001

max_epoch = 2 # 自行设置训练轮数
num_val = 1  # 经过多少轮进行验证

best_val_acc = 0
best_epoch = 0

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 2 * 2, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 64 * 2 * 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# data input
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_losses = []
train_acces = []


def train(model):
    for epoch in range(max_epoch):
        timestart = time.time()
        train_loss = 0
        total_num=0
        acc_epoch = 0
        total_epoch=0
        model.train()
        for i, data in enumerate(trainloader, 0):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            outs = model(imgs)
            loss = loss_function(outs, labels)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item() * imgs.size(0)  # imgs.size(0)批次
            total_num += imgs.size(0)
            _, prediction = torch.max(outs.data, 1)  # Alex_Net

            # train_acc += torch.sum(prediction == labels)
            # print(train_acc.cpu().item(), end=' ')

        # if epoch % num_val == 0:
        #     correct = 0  # 预测正确的样本数量
        #     total = 0  # 总共样本数量
        #     accuracy = 0
        #     with torch.no_grad():  # 验证的时候不需要计算梯度（避免产生计算图）
        #         for data in testloader:  # 获取一个批次的样本
        #             images, labels = data
        #             images = images.to(device)  # 将images展开为维度为784的向量
        #             labels = labels.to(device)
        #             outputs = model(images)
        #             _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 指返回的是outputs.data每行最大值的索引
        #             total += labels.size(0)
        #             correct += (predicted == labels).sum().item()  # 张量之间的比较运算
        #     accuracy = correct / total
        #     train_acces.append(accuracy)

            if (i + 1) % 100 == 0:
                correct = 0  # 预测正确的样本数量
                total = 0  # 总共样本数量
                accuracy = 0
                with torch.no_grad():  # 验证的时候不需要计算梯度（避免产生计算图）
                    for data in testloader:  # 获取一个批次的样本
                        images, labels = data
                        images = images.to(device)  # 将images展开为维度为784的向量
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 指返回的是outputs.data每行最大值的索引
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()  # 张量之间的比较运算
                        total_epoch += labels.size(0)
                        acc_epoch += (predicted == labels).sum().item()  # 张量之间的比较运算
                accuracy = correct / total
                train_acces.append(accuracy)
                print('Epoch {}, Step {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, i + 1, loss.item(),accuracy))

        print('Epoch {}, Average Loss: {:.4f}, Average Accuracy: {:.4f}'.format(epoch + 1,train_loss/total_num, acc_epoch/total_epoch))
        print('epoch %d cost %3f sec' % (epoch + 1, time.time()-timestart))

def Visualize(losses,acces):
    plt.subplot(2, 1, 1)
    plt.grid(True, linestyle='--', alpha=0.5)  # 默认是True，风格设置为虚线，alpha为透明度
    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('traing loss')
    plt.savefig('./plt_png/loss.png')
    # plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(acces, label="Train Acc")
    plt.xlabel('step')
    plt.ylabel('accuracy')
    plt.savefig('./plt_png/acc.png')
    plt.legend()

    plt.show()


def test(model):
    correct = 0
    total = 0
    model.eval()
    for data in testloader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy : %d %%' % (100*correct/total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':

    # model
    # model = BP_model()
    # model = LeNet5() # 用哪个模型，打开哪个，屏蔽另外两个
    model = CNN()
    model.to(device) # 选择cpu或者gpu
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train
    print('training:')
    timestart = time.time()
    train(model)
    print('finished training, cost %3f sec' % (time.time() - timestart))
    test(model)
    Visualize(train_losses,train_acces)



