"""把训练好的模型导出为 ONNX，用于在 netron.app 中核对网络结构。

用法：python 导出onnx.py --model derainnet

导出时必须喂一个假输入，因为 PyTorch 是动态图，模型真正的结构要跑一遍
才知道；这个输入只决定图的形状（batch=1, 3 通道, 128×128），数值无关。

导出后应能在 netron.app 中看到三分支结构对应的节点：
MaxPool ×2（下采样）、Resize ×3（上采样，ONNX 里 Upsample 记为 Resize）、
Concat ×1（三支拼接），输入输出都是 [1, 3, 128, 128]。
"""

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
    model.eval()          # 切到推理模式：关掉 Dropout、固定 BatchNorm，
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
