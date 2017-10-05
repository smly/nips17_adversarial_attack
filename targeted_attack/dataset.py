# -*- coding: utf-8 -*-
import glob
from pathlib import Path

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class LeNormalize(object):
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


def original_transform(sz):
    tf = transforms.Compose([
        transforms.Scale(sz),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
    ])
    return tf


def default_inception_transform(sz):
    tf = transforms.Compose([
        transforms.Scale(sz),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        LeNormalize(),
    ])
    return tf


class DataLoader(data.DataLoader):
    def get_filenames(self, idx, size):
        idx_st = idx * self.batch_size

        return [
            self.dataset.get_filename(file_idx)
            for file_idx in range(
                idx_st,
                idx_st + size)]


class Dataset(data.Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.transform = transform

        self.imgs = sorted(list(glob.glob(str(
            Path(input_dir) /
            Path("./*.png")))))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def set_transform(self, transform):
        self.transform = transform

    def get_filename(self, idx):
        return Path(self.imgs[idx]).name
