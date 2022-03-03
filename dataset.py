import os
from PIL import Image
import numpy as np
import torch
def read_font_images(font_dir, is_train=True):
    """读取所有font图像并标注。"""
    txt_fname = os.path.join(font_dir, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        feature = Image.open(os.path.join(
            font_dir, 'JPEGImages', f'{fname}.jpg')).convert("1")
        features.append(np.array(feature.copy()))
        feature.close()
        label = Image.open(os.path.join(
            font_dir, 'SegmentationClassAug', f'{fname}.png'))
        labels.append(np.array(label.copy()))
        label.close()
    return features, labels

# 字体数据集
class FontSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, font_dir):
        self.features, self.labels = read_font_images(font_dir, is_train)

    def __getitem__(self, idx):
        p1 = 288 - self.labels[idx].shape[0]
        p2 = 288 - self.labels[idx].shape[1]
        label_pad = np.pad(self.labels[idx], ((p1//2, p1 - p1//2),
                           (p2//2, p2 - p2//2)), 'constant', constant_values=0)
        feature_pad = np.pad(self.features[idx], ((
            p1//2, p1 - p1//2), (p2//2, p2 - p2//2)), 'constant', constant_values=1)
        return (torch.from_numpy(feature_pad).reshape([1, 288, 288]).float(), torch.from_numpy(label_pad).reshape([288, 288]).long())

    def __len__(self):
        return len(self.features)

if __name__ =="__main__":
    TrainDataset = FontSegDataset(True,"data/标准宋体")
    f,l = TrainDataset[5]
    np.set_printoptions(threshold  =  1e6 ) #设置打印数量的阈值
    # print(f.shape)
    # print(l.shape)