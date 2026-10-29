"""简单对比损失（Easy Contrastive Loss）。

像素损失比的是逐点差异，感知损失比的是特征图的逐位置差异；
对比损失把每张图压成一条特征向量，比的是两条向量的方向（余弦相似度）。

去雨任务里，一张预测图要同时对齐两件事：

    预测  与  无雨真值    →  方向要像，余弦相似度要大
    预测  与  有雨输入    →  方向要不像，余弦相似度要小

第二条用来防止模型把输入原样输出——那样第一条也能满足，但什么都没做。

"压成一条向量"用 AdaptiveAvgPool2d(1) 完成：不管特征图是多大，
都一次性平均成 1×1，每个通道只剩一个数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class EasyContrastiveLoss(nn.Module):
    """用 VGG19 中间层特征 + 余弦相似度计算的对比损失。

    输入三张 [0, 1] 范围的 RGB 张量，形状 (B, 3, H, W)，
    返回一个标量损失。
    """

    def __init__(self):
        super(EasyContrastiveLoss, self).__init__()
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].requires_grad_(False).eval()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, derained, clean, rainy):
        """三个输入，缺一不可。

        derained  模型输出（去雨结果）——唯一需要梯度的那个
        clean     真实无雨图——常量
        rainy     有雨输入图——常量
        """

        # 只有 derained 需要梯度；clean / rainy 是常量，不建计算图省显存
        derained = (derained-self.mean)/self.std
        clean = (clean-self.mean)/self.std
        rainy = (rainy-self.mean)/self.std
        derained = self.vgg(derained)
        with torch.no_grad():
            clean = self.vgg(clean)
            rainy = self.vgg(rainy)
        derained = self.pool(derained).flatten(1)
        clean = self.pool(clean).flatten(1)
        rainy = self.pool(rainy).flatten(1)
        sim1 = -F.cosine_similarity(derained, clean, dim=1)
        sim2 = F.cosine_similarity(derained, rainy, dim=1)
        sim = (1*sim1+1*sim2).mean()
        return(sim)
