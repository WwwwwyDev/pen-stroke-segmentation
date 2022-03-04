import numpy as np

def Pixel_Accuracy_Class(confusion_matrix):
    Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    Acc[Acc == 0] = np.nan
    # print(Acc)
    Acc = np.nanmean(Acc)
    return Acc