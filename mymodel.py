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


# ============================================================================
# 下面是我为「附加1」搭的模型，和上面的 MyModel 是两个东西：
# MyModel 是任务一用来跑通通路的最小对照，RainNet 是附加1 要求的「自己的模型」。
# ============================================================================


class ResBlock(nn.Module):
    """基础残差块：卷积 → 激活 → 卷积，再把输入加回输出。

    设计思路：附加1 要训 200 轮，纯卷积一路堆下去容易出现梯度衰减，
    残差的那条恒等映射让梯度能直接回传，也让每一块只需要学"在输入上改多少"，
    而不是"从零画出特征"。

    作用：RainNet 里所有特征提取和重建都由它完成，只负责维持通道数和分辨率不变。

    参考：ResNet 的恒等映射；PReNet（arXiv:1901.09221）里的递归残差块
    也是同一个做法，它的 PRN 主干就是若干个这样的块堆起来的。
    """

    def __init__(self, ch):
        super(ResBlock, self).__init__()
        # 两层 3×3 卷积夹一个 ReLU，输入输出的通道数和分辨率都不变
        self.model = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        # 残差：让这一块只学"在输入上改多少"，而不是从零画出特征
        y = x
        x = self.model(x)
        x = x + y
        return x


class RainNet(nn.Module):
    """附加1 的模型：三级编解码 + 三个不同尺度的雨头加权融合。

    设计思路：
      一是沿用「预测雨层」而不是「直接预测干净图」，最后一步做输入减雨层。
        这个写法任务一的 MyModel、任务二的 DerainNet 都在用，好处是网络
        只需要学雨这个稀疏的东西，背景直接抄输入。
      二是编解码骨架负责在三种分辨率上看图，浅层看细节、深层看走向。
      三是解码器每一级都伸出一个雨头，各自出一版雨层，再用三个可学习标量
        加权求和，让网络自己决定该信哪个尺度。这一条是参考 SPANet
        （arXiv:1904.01538）的多尺度+空间注意力思想，但把注意力退化成
        可学习权重，因为注意力模块依赖四方向 IRNN 提供的全局上下文，
        单独抠出来拿不到那个输入。

    作用：替代 MyModel，作为一个参数量在 4M 以内、能在 Rain200L 上训练的
    去雨模型，用来回答附加1 第 5 问。

    参考：PReNet（arXiv:1901.09221）的残差雨层与残差块；
    SPANet（arXiv:1904.01538）的多尺度检测思想。
    """

    def __init__(self):
        super(RainNet, self).__init__()

        # ---- 公共部件 ----
        self.conv_in = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 池化只改分辨率不改通道，所以每次下采样后要补一个卷积把通道翻倍
        self.down1 = nn.Conv2d(32, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)

        # ---- 编码器：分辨率减半、通道翻倍 ----
        self.enc1 = nn.Sequential(ResBlock(32), ResBlock(32))          # 128×128
        self.enc2 = nn.Sequential(ResBlock(64), ResBlock(64))          #  64×64
        self.enc3 = nn.Sequential(ResBlock(128), ResBlock(128))        #  32×32
        self.bottleneck = nn.Sequential(ResBlock(128), ResBlock(128))  #  16×16

        # ---- 解码器：先把跳连拼进来，再用卷积把通道压回去 ----
        # 括号里是拼接后的通道数 = 上采样后的通道 + 跳连的通道
        self.bridge3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.bridge2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.bridge1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(ResBlock(32), ResBlock(32))

        # ---- 三个雨头：各自把本级的特征压成 3 通道雨层 ----
        self.head1 = nn.Conv2d(128, 3, 3, padding=1)   # 32×32
        self.head2 = nn.Conv2d(64, 3, 3, padding=1)    # 64×64
        self.head3 = nn.Conv2d(32, 3, 3, padding=1)    # 128×128

        # 三个尺度雨层的融合权重。写成 nn.Parameter 它才会进优化器一起学
        self.w = nn.Parameter(torch.ones(3))

        # 三个雨头零初始化：这样第 0 步雨层恒等于 0，输出严格等于输入，
        # 训练从「什么都不做」的 PSNR 起跑，而不是从 15 dB 往回爬。
        # 理由详见 Reports/Part2-第一部分-任务二.md 第 1.6 节。
        for h in (self.head1, self.head2, self.head3):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, x):
        # 形状对照表（batch 按 8 写，实际是多少都行）：
        #
        #   变量   来源                          形状              通道
        #   f      conv_in(x)                    8,  32, 128, 128     32
        #   s1     enc1(f)                       8,  32, 128, 128     32
        #   s2     enc2(pool(s1))                8,  64,  64,  64     64
        #   s3     enc3(pool(s2))                8, 128,  32,  32    128
        #   b      bottleneck(pool(s3))          8, 128,  16,  16    128
        #   ─────────────────────────────────────────────────────────
        #   t3     cat(up(b),  s3)               8, 256,  32,  32    256
        #   d3     dec1(bridge3(t3))             8, 128,  32,  32    128
        #   r1     head1(d3)                     8,   3,  32,  32      3
        #   t2     cat(up(d3), s2)               8, 192,  64,  64    192
        #   d2     dec2(bridge2(t2))             8,  64,  64,  64     64
        #   r2     head2(d2)                     8,   3,  64,  64      3
        #   t1     cat(up(d2), s1)               8,  96, 128, 128     96
        #   d1     dec3(bridge1(t1))             8,  32, 128, 128     32
        #   r3     head3(d1)                     8,   3, 128, 128      3
        #   ─────────────────────────────────────────────────────────
        #   rain   w[0]*up(up(r1)) + w[1]*up(r2) + w[2]*r3
        #                                        8,   3, 128, 128
        #   输出   x - rain                      8,   3, 128, 128

        # 注意 s2 是从 s1 池化来的，s3 从 s2 来，不是三个都从 f 来
        f = self.conv_in(x)
        s1 = self.enc1(f)
        s2 = self.enc2(self.down1(self.pool(s1)))
        s3 = self.enc3(self.down2(self.pool(s2)))
        b = self.bottleneck(self.pool(s3))

        # 解码：每一级把上一级的结果上采样后和同级跳连拼起来
        d3 = self.dec1(self.bridge3(torch.cat([self.up(b), s3], dim=1)))
        r1 = self.head1(d3)
        d2 = self.dec2(self.bridge2(torch.cat([self.up(d3), s2], dim=1)))
        r2 = self.head2(d2)
        d1 = self.dec3(self.bridge1(torch.cat([self.up(d2), s1], dim=1)))
        r3 = self.head3(d1)

        # 三个尺度的雨层都放大到 128×128 再按学到的权重加权
        rain = self.w[0] * self.up(self.up(r1)) + self.w[1] * self.up(r2) + self.w[2] * r3
        return x - rain