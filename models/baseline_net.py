import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """保持形状不变的残差块：两层的通道数都是 64，空间尺寸也不变。

    结构为 卷积 → ReLU → 卷积，结果与输入相加后再激活。

    这里的 padding=1 是必需的而不是习惯写法：3×3 卷积每层会让
    特征图缩小 2，不加 padding 的话四个 Block 串下来就与输入形状对不上，
    最后一步"原图减输出"会直接报维度错误。

    结构参考 ResNet（He et al., 2016）中的 BasicBlock，去掉了下采样分支，
    只保留恒等映射那一支。
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + y
        x = self.relu(x)
        return x


class BaselineNet(nn.Module):
    """去雨基线模型：入口卷积 → 四个残差块 → 出口卷积 → 与输入相减。

    中间一共 8 个卷积（4 个 Block × 每块 2 层），入口和出口的两个卷积
    不计入这 8 层。参数量 298,947，约 0.30 M。

    出口卷积输出的是"雨层"而非干净图，最终结果用 有雨图 − 雨层 得到。
    这样网络只需要学习稀疏的雨纹，不必重建整张图像，收敛更容易。
    两种解释的取舍见 Reports/Part1-任务一.md。
    """

    def __init__(self):
        super(BaselineNet, self).__init__()
        self.convin = nn.Conv2d(3, 64, 3, padding=1)
        self.model = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )
        self.convout = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        y = x
        x = self.convin(x)
        x = self.model(x)
        x = self.convout(x)
        x = y - x
        return x
