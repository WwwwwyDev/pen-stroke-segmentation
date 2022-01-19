import torch
from torch import nn
from models.mynet import mynet
from dataset import FontSegDataset


DATA_BASE_URL = "data/CCSSD/DATA_GB6763_SS/SS2017"
MODEL_NAME = "mynet-DATA_GB6763_SS.pt"
BATCH_SIZE = 16
EPOCHS = 80
IS_USE_GPU = True
GPU_DEVICE = 0


if __name__ == '__main__':
    TrainDataset = FontSegDataset(True, DATA_BASE_URL)
    batch_size = BATCH_SIZE
    # 定义数据集迭代器
    train_iter = torch.utils.data.DataLoader(
        TrainDataset, batch_size, shuffle=True, drop_last=True)
    print("1.数据集加载成功")
    # 定义网络
    net = mynet(35)
    print("2.网络定义成功")
    if not IS_USE_GPU:
        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(), lr=0.0002)
        counter = 0   #计数器
        epochs = EPOCHS
        # train
        print("3.开始训练")
        for epoch in range(epochs):
            print('training_epoch', epoch+1, "of", epochs)
            for X, Y in train_iter:
                Y_hat = net(X.float())
                loss = loss_function(Y_hat, Y.long())
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                counter += 1
                if (counter % 100 == 0):
                    print("counter = ", counter, "loss = ", loss.item())
            torch.save(net, 'checkpoint/'+MODEL_NAME)
        print("训练结束")
    else:
        net = net.cuda(GPU_DEVICE)
        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(), lr=0.0002)
        counter = 0   #计数器
        epochs = EPOCHS
        # train
        print("3.开始训练")
        for epoch in range(epochs):
            print('training_epoch', epoch+1, "of", epochs)
            for X, Y in train_iter:
                Y_hat = net(X.float().cuda(GPU_DEVICE))
                loss = loss_function(Y_hat, Y.long().cuda(GPU_DEVICE))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                counter += 1
                if (counter % 100 == 0):
                    print("counter = ", counter, "loss = ", loss.item())
            torch.save(net, 'checkpoint/'+MODEL_NAME)
        print("训练结束")