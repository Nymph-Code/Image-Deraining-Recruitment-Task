# ══════════════════════════════════════════════════════════════
#  Part2 附加1 硬性要求之一：用 ONNX 导出模型，在 netron.app 里看结构
# ══════════════════════════════════════════════════════════════
#
#  ── 什么是 ONNX ────────────────────────────────────────────
#
#  PyTorch 的模型是一堆 Python 代码 + 张量，只能在 PyTorch 里跑。
#  ONNX（Open Neural Network Exchange）是一种【通用格式】，
#  可以把模型存成一个 .onnx 文件，任何支持 ONNX 的框架都能加载：
#  TensorFlow、ONNX Runtime、手机端、浏览器端……
#
#  导出之后用 https://netron.app/ 打开这个文件，
#  就能看到整个网络的图形结构：每一层是什么、输入输出多大、怎么连。
#
#  ── 为什么要用 torch.randn 造一个假输入 ────────────────────
#
#  PyTorch 是【动态图】：模型真正的结构要跑一遍才知道。
#  所以导出时必须喂一个样本，让它把整条 forward 走一次，
#  沿途记下每一步干了什么，才能翻译成静态的 ONNX 图。
#
#  这个假输入只决定【形状】（batch=1, 3 通道, 128×128），
#  数值是多少完全不重要，模型参数也不受影响。
#
#  ── 导出后要检查什么 ──────────────────────────────────────
#
#  1. 文件生成了没有，多大
#  2. netron.app 里打开，能找到这三个特征节点：
#        MaxPool  ×2      （下采样）
#        Resize   ×3      （上采样，ONNX 里 Upsample 叫 Resize）
#        Concat   ×1      （三支拼接）
#  3. 输入输出都是 [1, 3, 128, 128]
#
#  能找到这三个，说明你的三分支结构真的被完整导出了，
#  而不只是一个"看起来像"的空壳。
# ══════════════════════════════════════════════════════════════

import os
import torch

from models.derainnet import DerainNet
from models.baseline_net import BaselineNet
from models.mymodel import MyModel


MODELS = {
    "derainnet": DerainNet,
    "baseline": BaselineNet,
    "mymodel": MyModel,
}


def export(model_name, out_dir="onnx"):
    if model_name not in MODELS:
        raise ValueError(f"未知模型 {model_name}，可选：{list(MODELS)}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = MODELS[model_name]()
    model.eval()          # 切到推理模式：关掉 Dropout、固定 BatchNorm
                          # 导出的是推理用的图，不是训练用的图

    dummy = torch.randn(1, 3, 128, 128)
    path = os.path.join(out_dir, f"{model_name}.onnx")

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["rainy"],       # 给输入起个名字，netron 里会显示
        output_names=["derained"],   # 给输出起个名字
        opset_version=17,            # ONNX 算子集版本，17 兼容性好
        dynamo=False,                # 用传统 TorchScript 导出器，
                                     # 图更干净、更好在 netron 里读
    )

    size_mb = os.path.getsize(path) / 1024 / 1024
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] {path}")
    print(f"     参数量 {n_params:,}  文件大小 {size_mb:.2f} MB")
    print(f"     打开 https://netron.app/ ，把这个文件拖进去")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="derainnet", choices=list(MODELS))
    args = parser.parse_args()
    export(args.model)
