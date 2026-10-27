import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════
#  第一步：定义一个 Residual Block 类
# ══════════════════════════════════════════════════════════════
#
#  写在 BaselineNet 上面。继承 nn.Module，写法和 BaselineNet 一样。
#
#  ── __init__ 里要准备什么 ──────────────────────────────────
#
#  1. 两个卷积层，规格：
#       输入通道 64 → 输出通道 64
#       卷积核 3×3
#       padding = 1
#
#     想一想：为什么 padding 必须是 1？
#     （提示：3×3 的卷积核扫过 128 宽的图，不加 padding 会变成多宽？
#       再想：如果每层都缩小 2，4 个 Block 之后图还剩多大？）
#
#  2. 激活函数，用 nn.ReLU()
#     题目没规定用什么，ReLU 是默认选择。想换别的可以，但要能说出理由。
#
#  ── forward 里按顺序做四件事 ──────────────────────────────
#
#  1. 输入 → 第一个卷积 → 激活
#  2. 上一步的结果 → 第二个卷积        （这一步后面先不激活）
#  3. 第 2 步的结果 + 【forward 的参数 x】   ← 残差相加
#  4. 对相加的结果做激活 → 返回
#
#  ── 最常见的错误 ──────────────────────────────────────────
#
#  第 3 步加的必须是「进这个 Block 之前的原始 x」，
#  不是中间结果，也不是"卷积后的结果加自己"。
#  写成后者的话，网络里压根没有残差，等于白写。
#
#  自检：Block 的输入输出形状必须一模一样
#        (2, 64, 128, 128) 进去 → (2, 64, 128, 128) 出来

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64,64,3,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64,64,3,padding=1)

    def forward(self,x):
        y = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x+y
        x = self.relu(x)
        return(x)

        
# ══════════════════════════════════════════════════════════════
#  第二步：搭 BaselineNet 的整体结构
# ══════════════════════════════════════════════════════════════
#
#  __init__ 里准备三样东西，顺序不能乱：
#
#  1. 入口卷积：3 通道 → 64 通道，3×3，padding=1
#       输入是 RGB 三通道图，这一步把"3 种颜色"变成"64 种特征"。
#       后面所有 Block 都在 64 通道上工作，所以入口这一层不能省。
#
#  2. 四个 Residual Block，用 nn.Sequential 串起来
#       题目说的"8 层"，是 4 个 Block × 每个 Block 2 个卷积 = 8。
#       注意：入口和出口的两个卷积不算在这 8 层里。
#
#  3. 出口卷积：64 通道 → 3 通道，3×3，padding=1
#       最后要输出一张 RGB 图，所以通道数得回到 3。
#
#  ── 关键的设计决定（题目用"（重要）"标出来了）─────────────
#
#  出口卷积吐出来的这 3 个通道，代表什么？两种理解：
#
#     理解 A（直接学习）：它就是"去雨后的干净图"，模型一步到位
#     理解 B（残差学习）：它是"雨层"，干净图 = 有雨图 − 雨层
#
#  选哪个、为什么，去 Reports/Part1-任务一.md 里写清楚。
#  然后让 forward 的最后一步和你的选择一致。
#  评审会把这两边对着看，不一致的话这一部分等于白做。
#
#  如果你选 B，forward 最后要返回「输入图像 x 减去出口卷积的结果」。
#  选 A 则直接返回出口卷积的结果。


class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.convin = nn.Conv2d(3,64,3,padding=1)
        self.model = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )
        self.convout = nn.Conv2d(64,3,3,padding=1)
        # 1. 入口卷积：3 → 64，3×3，padding=1
        # 2. 四个 Residual Block
        # 3. 出口卷积：64 → 3，3×3，padding=1

    def forward(self, x):
        y = x
        x = self.convin(x)
        x = self.model(x)
        x = self.convout(x)
        x = y-x
        return(x)
        


# ══════════════════════════════════════════════════════════════
#  写完怎么自检
# ══════════════════════════════════════════════════════════════
#
#  在 repo 目录下执行：
#
#    "D:\Anaconda3\envs\derain\python.exe" -c "import torch; from models.baseline_net import BaselineNet; m=BaselineNet(); y=m(torch.rand(2,3,128,128)); print(y.shape, sum(p.numel() for p in m.parameters()))"
#
#  过关标准：
#
#   ① 输出形状是 torch.Size([2, 3, 128, 128])
#      不是这个形状，先回头查 forward 有没有漏掉某一步
#
#   ② 参数量在 0.29M ~ 0.30M 附近（算上 bias 会多一点）
#      手算核对：
#        中间 8 层卷积   64×64×3×3×8 = 294912
#        入口卷积        3×64×3×3    = 1728
#        出口卷积        64×3×3×3    = 1728
#        合计                         ≈ 0.298M
#      差得很远（比如 0.0x M 或 3M 以上），说明层数或通道数写错了
#
#   ③ 再验一次残差有没有真的加上：
#      把 Block 里的两个卷积故意换成"输出全 0"，如果残差写对了，
#      整个网络的输出应该和输入很接近；如果没写残差，输出就废了。
#      这一步不做也行，但做了你会对"残差在干什么"有实感。
