# -*- coding: utf-8 -*-
from pathlib import Path
import glob
import math

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


DEFAULT_CROP_PCT = 0.875


class LeNormalize(object):
    # Normalize to [-1.0, 1.0]
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


def default_inception_transform(sz):
    tf = transforms.Compose([
        transforms.Scale(sz),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        LeNormalize(),
    ])
    return tf


def default_transform(sz):
    tf = transforms.Compose([
        transforms.Scale(sz),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
    ])
    return tf


def default_transform_v2(sz):
    tf = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return tf


def default_transform_v3(sz):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    tf = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return tf


def transforms_eval(img_size=224, crop_pct=None):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    if crop_pct is None:
        if img_size == 224:
            scale_size = int(
                math.floor(img_size / DEFAULT_CROP_PCT)
            )
        else:
            scale_size = img_size
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    normalize = transforms.Normalize(
        mean=[124 / 255, 117 / 255, 104 / 255],
        std=[1 / (.0167 * 255)] * 3)

    return transforms.Compose([
        transforms.Scale(scale_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize])


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

        self.imgs = list(glob.glob(str(
            Path(input_dir) /
            Path("./*.png"))))

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
