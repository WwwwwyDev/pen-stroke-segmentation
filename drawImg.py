from sqlite3 import DatabaseError
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from models.mynet import mynet
from dataset import FontSegDataset
import matplotlib.pyplot as plt

MODEL_NAME = "mynet-标准宋体-50epochs.pt"
DATA_BASE_URL = "data/标准宋体"

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