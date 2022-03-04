from sqlite3 import DatabaseError
import torch
from torch import nn
from evaluate import fwiou, generate_matrix, miou , mpa
from models.mynet import mynet
from dataset import FontSegDataset
import matplotlib.pyplot as plt

MODEL_PATH ="checkpoint/segnet-标准宋体-10epochs.pt"
DATA_BASE_URL = "data/标准宋体"

if __name__ == '__main__':
    net = torch.load(MODEL_PATH, map_location='cpu')
    TestDataset = FontSegDataset(False, DATA_BASE_URL)
    miousum = 0
    mpasum = 0
    fwiousum = 0
    l = len(TestDataset)
    # l = 100
    for i in range(l):  
        X = TestDataset[i][0].unsqueeze(0)
        predict = net(X).argmax(dim=1).squeeze(0).numpy()
        origin = TestDataset[i][1].numpy()
        matrix =generate_matrix(origin,predict)
        mpasum += mpa.Pixel_Accuracy_Class(matrix)
        miousum += miou.Mean_Intersection_over_Union(matrix)
        fwiousum += fwiou.Frequency_Weighted_Intersection_over_Union(matrix)

        if i and i % 100 == 0:
            print(i)
    print("mpa:%f"%(mpasum/l))
    print("miou:%f"%(miousum/l))
    print("fwiou:%f"%(fwiousum/l))
    