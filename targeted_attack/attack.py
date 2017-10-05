# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import collections

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from incepresv2 import inceptionresnetv2

from scipy.misc import imsave, imresize
from dataset import (
    Dataset,
    DataLoader,
    LeNormalize,
    original_transform,
    default_inception_transform)


FN_MODEL = "./ensadv_inceptionresnetv2.pth"


class TargetAttack(object):
    def __init__(self,
                 model,
                 max_eps=16,
                 num_steps=10,
                 n_iter=1,
                 step_alpha=1.0):
        self.model = model
        self.max_eps = max_eps

        # basic iter count per model
        self.n_iter = n_iter

        self.eps = max_eps / 255.0

        # For calclate step alpha
        self.num_steps = num_steps
        self.step_alpha = self.eps / self.num_steps

        self.eps2 = 2.0 * max_eps / 255.0
        self.step_alpha2 = self.eps2 / self.num_steps

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.cuda()

    def generate(self, X, X_val, target_classes):
        X_var = Variable(X, requires_grad=True)

        y_with_bg = torch.from_numpy(np.array(target_classes))
        y_with_bg = y_with_bg.cuda()
        y_with_bg_var = Variable(y_with_bg)

        step_counter = 0
        while step_counter < self.n_iter * 2:
            zero_gradients(X_var)

            output = model(X_var)
            n_instance = len(output)

            batch_size = min(10, n_instance)
            n_classes = 1001
            output = output.view(batch_size, n_classes)

            # 1001 classes
            loss = self.loss_fn(output, y_with_bg_var)
            loss.backward()

            # Normalize and scale gradient
            normed_grad = self.step_alpha2 * torch.sign(
                X_var.grad.data)
            # Perturb current input image by normalized and scaled grad
            step_adv = X_var.data - normed_grad
            # Compare with original X to keep max_epsilon limit
            total_adv = step_adv - X
            total_adv = torch.clamp(total_adv, -self.eps2, self.eps2)
            # Apply total adv perturbation to orig images and clip
            X_adv = X + total_adv
            X_adv = torch.clamp(X_adv, -1.0, 1.0)
            X_var.data = X_adv
            step_counter += 1

        # value range: (-1, 1) -> (0, 1)
        for t in X_var.data:
            t.add_(1.0).div_(2.0)
        X_var.data = torch.clamp(X_var.data, 0.0, 1.0)

        # Verification
        return self.validate_and_clamp(X_var.data, X_val)

    def clamp_with_orig(self, X_adv, X_val):
        # X_adv: torch.cuda.FloatTensor of size 299x299
        #        [0, 1] range values
        # X_val: torch.cuda.FloatTensor of size 299x299
        #        [0, 1] range values
        total_adv = X_adv - X_val
        total_adv = torch.clamp(
            total_adv, -self.max_eps / 255.0, self.max_eps / 255.0)
        X_adv = X_val + total_adv
        X_adv = torch.clamp(X_adv, 0.0, 1.0)
        return X_adv

    def validate_and_clamp(self, X_adv, X_val):
        # X_adv: torch.cuda.FloatTensor of size 299x299
        #        [0, 1] range values
        # X_val: torch.cuda.FloatTensor of size 299x299
        #        [0, 1] range values
        X_adv = self.clamp_with_orig(X_adv, X_val)
        return X_adv.permute(0, 2, 3, 1).cpu().numpy()


def main():
    input_dir = sys.argv[1]
    output_name = sys.argv[2]
    max_epsilon = int(sys.argv[3])

    all_images_target_class_df = pd.read_csv(
        os.path.join(input_dir, 'target_class.csv'),
        names=['ImageName', 'ClassType'])
    all_images_target_class = dict(zip(
        all_images_target_class_df.ImageName.values,
        all_images_target_class_df.ClassType.values))

    # NN Model
    m = inceptionresnetv2(
        num_classes=1001,
        pretrained='imagenet+background')
    m = m.cuda()
    m.load_state_dict(torch.load(FN_MODEL))
    m.eval()

    # Dataset
    ds = Dataset(
        input_dir,
        transform=default_inception_transform(299))
    ds_orig = Dataset(
        input_dir,
        transform=original_transform(299))

    # DataLoader
    loader = DataLoader(ds, batch_size=10, shuffle=False)
    loader_orig = DataLoader(ds_orig, batch_size=10, shuffle=False)

    # Attacker
    attacker = TargetAttack(
        m,
        max_eps=max_epsilon,
        num_steps=1 * 4,
        n_iter=6)  # n_iter * n_models  = 9 * 2

    preds = []
    filenames = []
    loader_iter = iter(loader)
    loader_orig_iter = iter(loader_orig)
    for idx, (X, y) in enumerate(loader):
        X_, y_ = loader_orig_iter.next()

        filenames = loader.get_filenames(idx, len(X))
        X = X.cuda()
        X_ = X_.cuda()
        X_adv = attacker.generate(
            X, X_,
            [all_images_target_class[fn] for fn in filenames])

        for i, filename in enumerate(filenames):
            im = X_adv[i]
            im = np.clip(im * 255, 0, 255).astype(np.uint8)

            out_path = str(Path(output_name) / Path(filename))
            imsave(out_path, im, format='png')


if __name__ == '__main__':
    main()
