from sqlite3 import DatabaseError
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from models.mynet import mynet
from models.unet import Unet
from models.segnet import SegNet
from dataset import FontSegDataset
import matplotlib.pyplot as plt

MODEL_PATH ="checkpoint/mynet-标准宋体-50epochs.pt"
DATA_BASE_URL = "data/标准宋体"

if __name__ == '__main__':
    # idx = 2190
    idx = 0
    net = torch.load(MODEL_PATH, map_location='cpu')
    TestDataset = FontSegDataset(False, DATA_BASE_URL)
    X = TestDataset[idx][0].unsqueeze(0)
    predict = net(X).argmax(dim=1).squeeze(0).numpy()
    origin = TestDataset[idx][1].numpy()
    plt.subplot(1,2,1)
    plt.imshow(origin)
    plt.subplot(1,2,2)
    plt.imshow(predict)
    plt.show()