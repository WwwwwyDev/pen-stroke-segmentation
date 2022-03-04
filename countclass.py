from dataset import FontSegDataset
DATA_BASE_URL = "data/方正卡通简体"

if __name__ =="__main__":
    TrainDataset = FontSegDataset(False, DATA_BASE_URL)
    res = set()
    for i in range(len(TrainDataset)):
        if i % 100 == 0:
            print(i)
            print(res)
            print(len(res))
        _, l = TrainDataset[i]
        l = l.numpy()
        for x in range(288):
            for y in range(288):
                res.add(l[x][y])
    print(res)
    print(len(res))