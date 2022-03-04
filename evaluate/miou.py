import numpy as np


def Mean_Intersection_over_Union(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    MIoU[MIoU == 0] = np.nan
    print(MIoU)
    MIoU = np.nanmean(MIoU) #跳过0值求mean,shape:[21]
    return MIoU