import torch
import torch.nn as nn
from models.baseline_net import ResidualBlock


# ══════════════════════════════════════════════════════════════
#  Part2 任务二：实现真正的 DeRainNet（三分支多尺度）
# ══════════════════════════════════════════════════════════════
#
#  ── 这次要写成什么样 ──────────────────────────────────────
#
#  和 BaselineNet 的"独栋"不同，这个模型同时用三种尺度看同一张图：
#
#      分支0  原图大小  128×128  →  自己就能看到近处细节
#      分支1  缩小 1÷2   64×64   →  一个像素对应原图 2×2 区域
#      分支2  缩小 1÷4   32×32   →  一个像素对应原图 4×4 区域
#
#  三支各用 4 个 ResidualBlock 处理，处理完都放大回 128×128，
#  沿通道拼起来（Concat），再用卷积融合成一个 3 通道的"雨层"。
#
#  为什么要这样？——这是任务二的思考题，写代码时先体会，
#  写完再回去用文字回答。
#
#  ── 期望达到的结果 ────────────────────────────────────────
#
#  ① 形状不变
#         输入 (B, 3, 128, 128)  →  输出 (B, 3, 128, 128)
#         任何一步形状对不上，就会在后面"原图 − 雨层"那里报错
#
#  ② 参数量约 1.0 M
#         入口        1,792
#         三支 Block  886,272   （73,856 × 4 个 Block × 3 支）
#         融合卷积    112,387   （192→64 的 110,656 + 64→3 的 1,731）
#         ─────────────────────
#         合计     1,000,451   ≈ 1.0 M
#
#         ★ 如果你的数字差得很远（比如只有 0.3M 或超过 4M），
#           说明分支数、Block 数或融合层的通道写错了
#
#  ③ 训练效果应当优于 baseline
#         baseline 训 100 轮：PSNR 34.94
#         这个模型只训 50 轮，目标是不低于 34.8
#         如果明显更差，先查三支的 Concat 维度和上采样倍数
#
#  ── 自检命令 ──────────────────────────────────────────────
#
#  python -c "import torch; from models.derainnet import DerainNet; m=DerainNet(); y=m(torch.rand(2,3,128,128)); print(y.shape, sum(p.numel() for p in m.parameters()))"
#
#  期望输出：torch.Size([2, 3, 128, 128]) 1000451
#
#  ── 注意 ──────────────────────────────────────────────────
#
#  · ResidualBlock 直接从 baseline_net 导入，不要抄一遍
#  · ResidualBlock 是写死的 64→64，所以入口卷积必须输出 64 通道
#  · 分支内部的"＋"加的是【下采样之后、进 Block 之前】那张图，
#    不是加分支外的 f。形状必须一致才能相加
# ══════════════════════════════════════════════════════════════


class DerainNet(nn.Module):
    def __init__(self):
        super(DerainNet, self).__init__()
        self.convin = nn.Conv2d(3,64,3,padding=1)
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
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        self.convtotal = nn.Sequential(
            nn.Conv2d(192,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,3,3,padding=1),
        )
        # ── 融合层最后一个卷积【零初始化】──────────────────────
        #  为什么必须这么做：
        #    三支的输出都是「Block 处理结果 + 自己的 x」，本来数值就不小，
        #    拼成 192 通道后随便随机加权的输出会很大。
        #    实测：不置零时初始化雨层平均减掉 0.187，
        #          输出 PSNR 只有 15.0 dB —— 比"什么都不做"的 29.3 dB 差了 14 dB。
        #          这样训练是在从 15 dB 往回爬，50 轮根本爬不回来
        #          （实测 18 轮一直在 21~24 dB 之间震荡，第 1 轮 24.33 就是最高点）。
        #
        #  置零之后：雨层恒等于 0 → 输出 = 原图 → 第 0 步就站在 29.26 dB，
        #          和 baseline 同一起跑线，之后每一步都是在 29 之上往上加。
        #
        #  这也是残差学习的原意：先保证"什么都不做"是免费的，再让网络学"该减什么"。
        nn.init.zeros_(self.convtotal[2].weight)
        nn.init.zeros_(self.convtotal[2].bias)
        # ① 入口卷积：3 → 64，padding=1
        #    出来的 f 形状是 (B, 64, 128, 128)，三支都从它出发

        # ② 三支各自的 4 个 ResidualBlock
        #    建议用 nn.Sequential(*[ResidualBlock() for _ in range(4)])
        #    或用 nn.Sequential(ResidualBlock(), ResidualBlock(), ...) 展开写

        # ③ 下采样和上采样
        #    self.pool = nn.MaxPool2d(2)
        #    self.up   = nn.Upsample(scale_factor=2, mode='bilinear',
        #                            align_corners=False)

        # ④ 融合卷积：192 → 64 → 3
        #    192 = 三支拼起来的通道数 (64 × 3)

    def forward(self, x):
        #  y = x                       ← 存下原图，最后要用

        #  ── 入口 ──────────────────────────────────────────
        #  x = self.convin(x)          → (B, 64, 128, 128)

        #  ── 分支0：原尺寸 ─────────────────────────────────
        #  b0 = 分支0的4个Block处理(x) + x       → (B, 64, 128, 128)

        #  ── 分支1：1/2 ────────────────────────────────────
        #  x1 = self.pool(x)                     → (B, 64,  64,  64)
        #  b1 = 分支1的4个Block处理(x1) + x1      → (B, 64,  64,  64)
        #  b1 = self.up(b1)                      → (B, 64, 128, 128)

        #  ── 分支2：1/4 ────────────────────────────────────
        #  x2 = self.pool(x1)                    → (B, 64,  32,  32)
        #  b2 = 分支2的4个Block处理(x2) + x2      → (B, 64,  32,  32)
        #  b2 = self.up(self.up(b2))             → (B, 64, 128, 128)

        #  ── 拼接 + 融合 ───────────────────────────────────
        #  out  = torch.cat([b0, b1, b2], dim=1) → (B, 192, 128, 128)
        #  rain = self.fuse(out)                 → (B,   3, 128, 128)

        #  ── 残差相减 ──────────────────────────────────────
        #  return y - rain
        y = x
        x = self.convin(x)
        b0 = self.model1(x)+x
        x1 = self.pool(x)
        b1 = self.model2(x1)+x1
        b1 = self.up(b1)
        x2 = self.pool(x1)                    
        b2 = self.model3(x2) + x2      
        b2 = self.up(self.up(b2))
        out  = torch.cat([b0, b1, b2], dim=1) 
        rain = self.convtotal(out)
        return y - rain


