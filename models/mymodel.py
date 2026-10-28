import torch
import torch.nn as nn


class MyModel(nn.Module):
    """Part2 第一部分任务一的最小模型：入口卷积 → 出口卷积 → 与输入相减。

    不含任何 ResidualBlock，只有两个卷积，结构上等价于一个线性的
    雨层估计器，用来先验证"有雨图 → 减雨层 → 干净图"这条通路能跑通、
    输入输出形状一致（(B, 3, 128, 128) → (B, 3, 128, 128)）。
    参数量 3,523。

    它的作用是与后续带残差块的模型作对照，说明网络容量的贡献，
    结果见 Reports/Part2-第一部分-任务一.md。
    """

    def __init__(self):
        super(MyModel, self).__init__()
        self.convin = nn.Conv2d(3, 64, 3, padding=1)
        self.convout = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        y = x
        x = self.convin(x)
        x = self.convout(x)
        x = y - x
        return x
