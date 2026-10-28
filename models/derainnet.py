import torch
import torch.nn as nn
from models.baseline_net import ResidualBlock


class DerainNet(nn.Module):
    """三分支多尺度去雨网络。

    与 BaselineNet 只用单一尺度不同，这里让同一张图在三种尺度上各处理一遍：

        分支 0   原尺寸 128×128   一个像素对应原图 1×1 区域
        分支 1   缩小 1/2  64×64   一个像素对应原图 2×2 区域
        分支 2   缩小 1/4  32×32   一个像素对应原图 4×4 区域

    三支各用 4 个 ResidualBlock 处理，结果加回各自的输入（残差），
    再上采样回 128×128，沿通道拼成 192 维，最后由两个卷积融合成
    3 通道的"雨层"，用原图相减去雨。

    这样做的理由是雨纹的尺度跨度很大：细密的雨丝在原始分辨率上
    最清楚，而大片的雨雾要在更粗的尺度上才能整体感知。单尺度网络
    必须在同一层里兼顾两种频率，多尺度分支把这件事拆开。

    融合结构参考了超分与去噪中常见的多尺度金字塔做法；分支内的
    残差块沿用 ResNet（He et al., 2016）的 BasicBlock。

    参数量 1,000,451，输入输出形状均为 (B, 3, 128, 128)。
    """

    def __init__(self):
        super(DerainNet, self).__init__()
        self.convin = nn.Conv2d(3, 64, 3, padding=1)
        self.model1 = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )
        self.model2 = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )
        self.model3 = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.convtotal = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

        # 融合层的最后一个卷积做零初始化。
        #
        # 三支的输出都是「Block 结果 + 各自的输入」，数值本来就不小，
        # 拼成 192 通道后若按默认方式随机初始化，融合出的"雨层"幅度会很大，
        # 减掉之后输出就废了。
        #
        # 实测（前 100 张训练图）：不置零时初始化的雨层平均减掉 0.187，
        # 输出 PSNR 只有 15.0 dB，比"什么都不做"的 29.3 dB 还差 14 dB，
        # 训练相当于从 15 dB 往回爬，50 轮爬不回来（18 轮一直在 21~24 dB
        # 之间震荡，第 1 轮的 24.33 就是最高点）。
        #
        # 置零之后雨层恒等于 0，输出 = 原图，第 0 步就站在 29.26 dB，
        # 与 baseline 同一起跑线，之后每一步都是在 29 之上往上加。
        #
        # 这也是残差学习的基本思路：先保证"什么都不做"是免费的，
        # 再让网络去学"该减掉什么"。
        nn.init.zeros_(self.convtotal[2].weight)
        nn.init.zeros_(self.convtotal[2].bias)

    def forward(self, x):
        y = x                                   # 存下原图，最后相减用

        x = self.convin(x)                      # (B, 64, 128, 128)

        # 分支 0：原尺寸
        b0 = self.model1(x) + x                 # (B, 64, 128, 128)

        # 分支 1：1/2 尺度，处理完升回原尺寸
        x1 = self.pool(x)                       # (B, 64,  64,  64)
        b1 = self.model2(x1) + x1               # (B, 64,  64,  64)
        b1 = self.up(b1)                        # (B, 64, 128, 128)

        # 分支 2：1/4 尺度，处理完连续上采样两次
        x2 = self.pool(x1)                      # (B, 64,  32,  32)
        b2 = self.model3(x2) + x2               # (B, 64,  32,  32)
        b2 = self.up(self.up(b2))               # (B, 64, 128, 128)

        # 三支沿通道拼接后融合成雨层
        out = torch.cat([b0, b1, b2], dim=1)    # (B, 192, 128, 128)
        rain = self.convtotal(out)              # (B,   3, 128, 128)

        return y - rain
