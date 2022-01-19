import os
from PIL import Image
import numpy as np
import torch

def read_font_images(font_dir, is_train=True):
    """读取所有font图像并标注。"""
    txt_fname = os.path.join(font_dir, '../../train.txt' if is_train else '../../val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        feature = Image.open(os.path.join(
            font_dir, 'JPEGImagesjpg', f'{fname}.jpg')).convert("1")
        features.append(np.array(feature.copy()))
        feature.close()
        label = Image.open(os.path.join(
            font_dir, 'SegmentationClass', f'{fname}.png')).convert("RGB")
        labels.append(np.array(label.copy()))
        label.close()
    return features, labels

# 字体数据集
class FontSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, font_dir):
        self.features, self.labels = read_font_images(font_dir, is_train)
        font_colormap = [[0, 0, 0], [0, 0, 128], [0, 0, 64], [0, 128, 0], [0, 128, 128],
                         [0, 128, 64], [0, 192, 0], [
                             0, 192, 128], [0, 64, 0], [0, 64, 128],
                         [128, 0, 0], [128, 0, 128], [128, 0, 64], [
            128, 128, 0], [128, 128, 128],
            [128, 192, 0], [128, 192, 128], [128, 64, 0], [
                128, 64, 128], [192, 0, 0], [192, 0, 128],
            [192, 128, 0], [192, 128, 128], [192, 192, 0], [
                192, 192, 128], [192, 64, 0], [192, 64, 128],
            [64, 0, 0], [64, 0, 128], [64, 128, 0], [64, 128, 128], [64, 192, 0], [64, 192, 128], [64, 64, 0], [64, 64, 128]]
        self.colormap2label = np.zeros(256 ** 3, dtype=np.int64)
        for i, colormap in enumerate(font_colormap):
            self.colormap2label[(colormap[0] * 256 + colormap[1])
                                * 256 + colormap[2]] = i
        pass

    def font_label_indices(self, colormap):
        """将标签中的RGB值映射到它们的类别索引。"""
        colormap = colormap.astype(np.int32)
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
               + colormap[:, :, 2])
        return self.colormap2label[idx]

    def __getitem__(self, idx):
        label = self.font_label_indices(self.labels[idx])
        p1 = 288 - label.shape[0]
        p2 = 288 - label.shape[1]
        label_pad = np.pad(label, ((p1//2, p1 - p1//2),
                           (p2//2, p2 - p2//2)), 'constant', constant_values=0)
        feature_pad = np.pad(self.features[idx], ((
            p1//2, p1 - p1//2), (p2//2, p2 - p2//2)), 'constant', constant_values=255)
        return (torch.from_numpy(feature_pad).reshape([1, 288, 288])/255.0, torch.from_numpy(label_pad).reshape([288, 288]))

    def __len__(self):
        return len(self.features)