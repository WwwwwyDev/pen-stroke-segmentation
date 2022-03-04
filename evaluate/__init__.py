import numpy as np


def generate_matrix(gt_image, pre_image,num_class=35):
        mask = (gt_image >= 0) & (gt_image < num_class)#ground truth中所有正确(值在[0, classe_num])的像素label的mask
        
        label = num_class * gt_image[mask].astype('int') + pre_image[mask] 
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class, num_class)#21 * 21(for pascal)
        return confusion_matrix