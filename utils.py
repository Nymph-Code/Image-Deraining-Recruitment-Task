import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image


# ══════════════════════════════════════════════════════════════
#  附加题 3.3：手写 PSNR
# ══════════════════════════════════════════════════════════════
#
#  目标：写一个函数，能顶替 skimage.metrics.peak_signal_noise_ratio
#
#  ── 接口（必须和 skimage 对得上，不然调用会报错）────────────
#
#      my_psnr(img1, img2, data_range=1.0)  ->  float
#
#      img1 / img2 : numpy 数组，形状 (128, 128, 3)，值域 [0, 1]
#      data_range  : 像素的最大取值，这里传 1.0
#
#  ── 要做的三件事 ──────────────────────────────────────────
#
#  1. 算 MSE
#        两张图逐元素相减 → 平方 → 对【全部元素】求平均
#        三个坐标：批内第 i 张图、128 行、128 列、3 个通道
#        一共 128 × 128 × 3 = 49152 个数，一起平均
#        （不是分通道算完再平均，用 np.mean 一把过就对）
#
#  2. 套公式
#        PSNR = 10 * log10( data_range ** 2 / MSE )
#        data_range = 1.0 时，分子就是 1
#
#  3. 处理边界情况
#        两张图完全相同时 MSE = 0，log10(0) = inf，
#        而且 numpy 会刷一行 RuntimeWarning: divide by zero
#        你要决定 MSE 为 0 时返回什么，并且让它和 skimage 的行为一致
#        （提示：先跑一次 skimage 看它对相同图片返回什么，再决定）
#
#  ── 写完怎么验证 ──────────────────────────────────────────
#
#  1. 单独对比：拿同一对图，你的函数和 skimage 各算一次，数值应该相同
#
#  2. 整体回归：在 repo 目录下执行
#        python main.py --data_dir "./data/Rain200L" --model baseline --mode test --checkpoint "checkpoints/best_baseline.pth.tar"
#     checkpoint 是第 96 轮保存的，那一轮的 PSNR 是 33.9059，
#     换掉之后测试结果应该落在这个数附近（不会完全相等，因为数据加载顺序不同）

def my_psnr(img1, img2, data_range=1.0):
    MSE = np.mean((img1-img2)**2)
    if MSE == 0:
        return float("inf")
    PSNR = 10 * np.log10(data_range**2/MSE)
    return(PSNR)


def calculate_metrics(img1, img2, lpips_fn):

    img1_np = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img2_np = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)

    batch_psnr, batch_ssim = 0, 0
    for i in range(img1_np.shape[0]):
        batch_psnr += my_psnr(img1_np[i], img2_np[i], data_range=1.0)
        batch_ssim += ssim(img1_np[i], img2_np[i], data_range=1.0, multichannel=True, channel_axis=2, win_size=7)

    avg_psnr = batch_psnr / img1_np.shape[0]
    avg_ssim = batch_ssim / img1_np.shape[0]

    img1_lpips = img1.clone() * 2 - 1
    img2_lpips = img2.clone() * 2 - 1

    with torch.no_grad():
        lpips_score = lpips_fn(img1_lpips, img2_lpips).mean().item()

    return avg_psnr, avg_ssim, lpips_score


def save_checkpoint(state, folder="checkpoints", filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, f"{folder}/{filename}")


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def save_some_examples(model, loader, epoch, folder, device):
    import os
    model.eval()
    rainy, clean, filename = next(iter(loader))
    rainy, clean = rainy.to(device), clean.to(device)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with torch.no_grad():
        derained = model(rainy)

    comparison = torch.cat([rainy, derained, clean], dim=0)
    save_image(comparison, f"{folder}/comparison_epoch_{epoch}.png", nrow=rainy.shape[0])
    model.train()