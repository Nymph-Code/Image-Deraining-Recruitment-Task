import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DerainDataset(Dataset):
    def __init__(self, root_dir, is_train=True, image_size=128):
        self.root_dir = root_dir
        self.is_train = is_train

        data_folder = 'train' if is_train else 'test'
        self.rainy_dir = os.path.join(self.root_dir, data_folder, 'input')
        self.clean_dir = os.path.join(self.root_dir, data_folder, 'target')

        self.rainy_images = sorted(os.listdir(self.rainy_dir))

        if self.is_train:
            # 随机水平翻转不能放在这里的 Compose 里，原因见 __getitem__。
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(image_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.rainy_images)

    def __getitem__(self, idx):
        rainy_path = os.path.join(self.rainy_dir, self.rainy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.rainy_images[idx])

        rainy_image = Image.open(rainy_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')

        rainy_tensor = self.transform(rainy_image)
        clean_tensor = self.transform(clean_image)

        # 随机水平翻转必须对两张图同步做。
        #
        # 原先把它写在 Compose 里，然后 self.transform(rainy_image) 和
        # self.transform(clean_image) 各调一次；RandomHorizontalFlip 每次
        # 调用都独立掷随机数，于是大约一半的样本变成「有雨图没翻、干净图翻了」，
        # 输入和目标完全对不上。
        #
        # 实测（前 100 张训练图，有雨图与干净图的 PSNR）：
        #     带随机翻转   平均 21.73 dB   最差 8.67 dB
        #     去掉随机翻转 平均 28.08 dB   最差 20.47 dB
        # 差距全部来自这条错位。
        #
        # 下面用同一个随机数控制两张图，保证要么都翻、要么都不翻。
        if self.is_train and random.random() < 0.5:
            rainy_tensor = torch.flip(rainy_tensor, dims=[2])   # dims=2 是宽度方向
            clean_tensor = torch.flip(clean_tensor, dims=[2])

        return rainy_tensor, clean_tensor, self.rainy_images[idx]