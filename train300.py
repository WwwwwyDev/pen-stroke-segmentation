import torch
from torch import nn
from models.mynet import mynet
from dataset import FontSegDataset

IS_USE_GPU = True
GPU_DEVICE = 0

if __name__ == '__main__':
    TrainDataset = FontSegDataset(True, "data/标准宋体")
    batch_size = 16
    # 定义数据集迭代器
    train_iter = torch.utils.data.DataLoader(
        TrainDataset, batch_size, shuffle=True, drop_last=True)
    print("1.数据集加载成功")
    # 定义网络
    net = mynet(35)
    print("2.网络定义成功")
    if not IS_USE_GPU:
        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(), lr=0.0001)
        counter = 0   #计数器
        epochs = 300
        # train
        print("3.开始训练")
        for epoch in range(epochs):
            print('training_epoch', epoch+1, "of", epochs)
            for X, Y in train_iter:
                Y_hat = net(X)
                loss = loss_function(Y_hat, Y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                counter += 1
                if (counter % 100 == 0):
                    print("counter = ", counter, "loss = ", loss.item())
                if (counter % 10 == 0):
                    with open("checkpoint/losstest/loss.txt","a") as file:
                        file.write(loss.item()+",")
            if (epoch+1) % 50 == 0:
                torch.save(net, 'checkpoint/losstest/'+"s-%depochs.pt"%epoch)
        print("训练结束")
    else:
        net = net.cuda(GPU_DEVICE)
        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(), lr=0.0001)
        counter = 0   #计数器
        epochs = 300
        # train
        print("3.开始训练")
        for epoch in range(epochs):
            print('training_epoch', epoch+1, "of", epochs)
            for X, Y in train_iter:
                Y_hat = net(X.cuda(GPU_DEVICE))
                loss = loss_function(Y_hat, Y.cuda(GPU_DEVICE))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                counter += 1
                if (counter % 100 == 0):
                    print("counter = ", counter, "loss = ", loss.item())
                if (counter % 10 == 0):
                    with open("checkpoint/losstest/loss.txt","a") as file:
                        file.write(str(loss.item())+",")
            if (epoch+1) % 50 == 0:
                torch.save(net, 'checkpoint/losstest/'+"s-%depochs.pt"%(epoch+1))
        print("训练结束")