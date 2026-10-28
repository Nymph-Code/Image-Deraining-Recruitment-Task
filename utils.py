import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image


def my_psnr(img1, img2, data_range=1.0):
    """手写的 PSNR，用来替代 skimage.metrics.peak_signal_noise_ratio。

    接口与 skimage 保持一致：接收两张形状为 (H, W, C)、值域 [0, 1] 的
    numpy 数组，返回 float。

    计算分两步：先对全部元素求均方误差（不分通道、不按通道平均），
    再套 PSNR = 10 * log10(data_range^2 / MSE)。

    MSE 为 0（两张图完全相同）时 log10(0) 会是 -inf，numpy 还会刷
    divide by zero 警告。这里显式返回 inf，与 skimage 对相同图片的行为一致。
    """
    MSE = np.mean((img1 - img2) ** 2)
    if MSE == 0:
        return float("inf")
    PSNR = 10 * np.log10(data_range ** 2 / MSE)
    return PSNR


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
