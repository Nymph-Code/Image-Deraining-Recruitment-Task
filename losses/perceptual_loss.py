"""感知损失（Perceptual Loss）。

像素损失（L1、L2）是逐像素比较两张图，隐含假设每个像素彼此独立。
但人判断两张图像不像，看的是整体结构和纹理，不是某个像素差了多少。
感知损失的做法是：把两张图先送进一个预训练网络，比较它们在中间层
提取出的特征图，而不是比较原始像素。

为什么用 VGG19：它在 ImageNet 上训练过，中间层的特征图已经学会了
表达边缘、纹理、局部结构这类东西，正好对应"看起来像不像"。
比较特征图，等于在问"两张图在高层的语义上差多少"，而不是"每个点差多少"。

为什么截到前 35 层：VGG19 的 features 一共 37 层，更深的层特征更抽象、
空间分辨率也更低，对去雨这种需要保留细节的任务帮助不大，截掉还能省显存。

为什么用 L1 而不是 L2：L2 对大误差惩罚更重，容易把注意力集中在少数
差异大的位置；L1 更稳，也更符合"感知差异"这种不那么强调极端值的度量。
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    """用 VGG19 中间层特征计算的感知损失。

    输入两张 [0, 1] 范围的 RGB 张量，形状 (B, 3, H, W)，
    返回一个标量损失。
    """

    def __init__(self):
        super(PerceptualLoss, self).__init__()

        # VGG 在这里只当尺子用，不参与训练：冻结参数、切成推理模式、截到前 35 层
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].requires_grad_(False).eval()
        self.criterion = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, generated, ground_truth):
        """generated 是模型输出，ground_truth 是真实无雨图，两者都要算特征。"""

        # 先按 ImageNet 的均值和标准差归一化，让输入分布和 VGG 训练时一致
        generated = (generated-self.mean)/self.std
        ground_truth = (ground_truth-self.mean)/self.std
        generated = self.vgg(generated)
        ground_truth = self.vgg(ground_truth)
        loss1 = self.criterion(generated,ground_truth)
        return loss1
