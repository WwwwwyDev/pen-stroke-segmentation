# pen-stroke-segmentation

#### 介绍
针对当前汉字笔画提取存在识别精度低和笔画分割不清等问题, 提出了一种辅助字体设计的汉字字体笔画分割模型. 该模型可对汉字图像进行有效的笔画分割。方法引入跳跃结构加强了笔画结构信息损失; 针对字体语义信息丢失的问题, 将模型中最大池化改为空洞卷积, 利用空洞卷积减少下采样的信息损失, 提升分割精度.

#### 论文地址
[论文地址](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDAUTO&filename=DLMY202205015&uniplatform=NZKPT&v=Iu5O7u8Ge54PiIWXii5d02_IbkDVWZxMQRpGIp3SHVmXw2wJesFn2Nt_2fo48rpD)

#### 数据集
[百度网盘](https://pan.baidu.com/s/1ajN7HQkPbbPcmAuKJJXCNA?pwd=f7tU)  提取码: f7tU 

#### 预训练模型

[百度网盘](https://pan.baidu.com/s/1YZkR2ezE27ZshPHmsGtE5A)  提取码：fgpk

#### 网络结构

![输入图片说明](img/%E6%88%AA%E5%B1%8F2022-06-15%20%E4%B8%8B%E5%8D%888.06.43.png)

#### 分割效果
![输入图片说明](img/%E5%9B%BE11.jpg)

