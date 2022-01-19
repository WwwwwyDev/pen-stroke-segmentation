from sqlite3 import DatabaseError
import torch
from torch import nn
from models.mynet import mynet
from evaluate import IOUMetric
from dataset import FontSegDataset
import matplotlib.pyplot as plt

MODEL_NAME = "mynet-DATA_GB6763_SS.pt"
DATA_BASE_URL = "data/CCSSD/DATA_GB6763_SS/SS2017"

if __name__ == '__main__':
    net = torch.load('checkpoint/'+MODEL_NAME, map_location='cpu')
    TestDataset = FontSegDataset(False, DATA_BASE_URL)
    X = TestDataset[0][0].unsqueeze(0)
    predict = net(X).argmax(dim=1).squeeze(0).numpy()
    origin = TestDataset[0][1].numpy()
    plt.subplot(1,2,1)
    plt.imshow(origin)
    plt.subplot(1,2,2)
    plt.imshow(predict)
    plt.show()
    mIOU = IOUMetric(35)
    sumn = 0
    for i in range(len(TestDataset)):  
        X = TestDataset[i][0].unsqueeze(0)
        predict = net(X).argmax(dim=1).squeeze(0).numpy()
        origin = TestDataset[i][1].numpy()
        sumn += mIOU.evaluate(predict,origin)
        if i % 100 == 0:
            print(i)
    print(sumn/len(TestDataset))
    