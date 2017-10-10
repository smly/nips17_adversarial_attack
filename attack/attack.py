# -*- coding: utf-8 -*-
# Approach: Round-robin iterative FGSM
# * Models: DenseNet, ResNet152, IncepV3, IncepResV2, IncepV4, ResNext101
import sys
from pathlib import Path
import collections

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision.models.densenet import densenet161
from torchvision.models.resnet import resnet34
from models_fbresnet import fbresnet152
from torchvision.models.inception import Inception3, inception_v3
from models_incepv3 import inception_v3 as inception_v3_fullconv
from models_incepresv2 import inceptionresnetv2
from models_inceptionv4 import inceptionv4
from models_resnext import resnext101_64x4d
import torchvision.transforms as transforms

from scipy.misc import imsave, imresize
from dataset import (
    Dataset,
    DataLoader,
    LeNormalize,
    original_transform,
    default_transform,
    default_transform_v2,
    default_inception_transform,
)


FN_MODEL1 = "./densenet161-17b70270.pth"
# FN_MODEL2 = "./resnet34-333f7ec4.pth"
FN_MODEL2 = "./resnet152-c11d722e.pth"
FN_MODEL3 = "./inception_v3_google-1a9a5a14.pth"
FN_MODEL4 = "./inceptionresnetv2-d579a627.pth"
FN_MODEL5 = "./inceptionv4-97ef9c30.pth"

FN_MODEL6 = "./advinceptv3_fullconv_state.pth"
FN_MODEL7 = "./ens3incepv3_fullconv_state.pth"
FN_MODEL8 = "./ens4incepv3_fullconv_state.pth"
FN_MODEL9 = "./ensadv_inceptionresnetv2.pth"

FN_MODEL10 = "./resnext101_64x4d.pth"


class RoundRobinEnsemble(object):
    def __init__(self,
                 models,
                 max_eps=16,
                 num_steps=10,
                 n_iter=1,
                 step_alpha=1.0):
        self.models = models
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

    def generate(self, X, X2, X_val):
        X_var = Variable(X, requires_grad=True)
        X2_var = Variable(X2, requires_grad=True)

        n_instances = len(X)
        batch_size = min(10, n_instances)

        y = torch.zeros(batch_size)  # dummy
        y = y.cuda()
        y_var = Variable(y)

        y_with_bg = torch.zeros(batch_size)
        y_with_bg = y_with_bg.cuda()
        y_with_bg_var = Variable(y_with_bg)

        y_bg_ll = torch.zeros(batch_size)
        y_bg_ll = y_bg_ll.cuda()
        y_bg_ll_var = Variable(y_bg_ll)

        # ------------------------------------------
        # Initialize y by using best model, IncResV2
        zero_gradients(X2_var)
        m_best = self.models[3]
        output = m_best(X2_var)
        y_with_bg_var.data = output.data.max(1)[1].view(-1)
        y_bg_ll_var.data = output.data.min(1)[1].view(-1)
        y_var.data = output.data.max(1)[1].view(-1) - 1

        # ------------------------------------------
        # RR1 (224x224), X: range values (0, 1)
        # 1:DenseNet161, 2:ResNet34
        step_counter = 0
        while step_counter < self.n_iter:
            zero_gradients(X_var)

            for model_i, model in enumerate([self.models[0],
                                             self.models[1],
                                             self.models[9]]):
                output = model(X_var)
                loss = self.loss_fn(output, y_var)
                loss.backward()

                # Normalize and scale gradient
                normed_grad = self.step_alpha * torch.sign(
                    X_var.grad.data)
                # Perturb current input image by normalized and scaled grad
                step_adv = X_var.data + normed_grad
                # Compare with original X to keep max_epsilon limit
                total_adv = step_adv - X
                total_adv = torch.clamp(total_adv, -self.eps, self.eps)
                # Apply total adv perturbation to orig images and clip
                X_adv = X + total_adv
                X_adv = torch.clamp(X_adv, 0.0, 1.0)
                X_var.data = X_adv
            step_counter += 1

        # --------------------------------------------
        # Rescale intermediate result to 299x299 scale
        # src: X_var.data  (224x224), value range (0, 1)
        # dst: X2_var.data (299x299), value range (-1, 1)
        to_rescaled_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale(299),
            transforms.ToTensor(),
            LeNormalize(),
        ])
        for idx in range(len(X)):
            X2_var.data[idx] = to_rescaled_tensor(X_var.data[idx].cpu())
        X2_var.data = torch.clamp(X2_var.data, -1.0, 1.0)

        # ------------------------------------------
        # RR2 (299x299), X2: range values (-1, 1)
        # 3:IncepV3, 4:IncResV2, 5:IncepV4
        step_counter = 0
        while step_counter < self.n_iter:
            zero_gradients(X2_var)
            for idx, model in enumerate(self.models[2:5]):
                output = model(X2_var)
                if idx == 0:
                    # 3:IncepV3
                    # 1000 classes
                    loss = self.loss_fn(output, y_var)
                    loss.backward()
                else:
                    # 4:IncResV2, 5:IncepV4
                    # 1001 classes
                    loss = self.loss_fn(output, y_with_bg_var)
                    loss.backward()

                # Normalize and scale gradient
                normed_grad = self.step_alpha2 * torch.sign(
                    X2_var.grad.data)
                # Perturb current input image by normalized and scaled grad
                step_adv = X2_var.data + normed_grad
                # Compare with original X to keep max_epsilon limit
                total_adv = step_adv - X2
                total_adv = torch.clamp(total_adv, -self.eps2, self.eps2)
                # Apply total adv perturbation to orig images and clip
                X_adv = X2 + total_adv
                X_adv = torch.clamp(X_adv, -1.0, 1.0)
                X2_var.data = X_adv
            step_counter += 1

        # ------------------------------------------
        # RR3 (299x299), X2: range values (-1, 1)
        # 6:AdvIncepV3, 7:Ens3AdvIncepV3,
        # 8:Ens4AdvIncepV3, 9:EnsAdvIncResV2
        step_counter = 0
        while step_counter < self.n_iter * 2:
            zero_gradients(X2_var)

            # 299x299 scale
            # AdvTrn
            for model in self.models[5:9]:
                output = model(X2_var)
                n_instances = len(output)

                batch_size = min(10, n_instances)
                n_classes = 1001
                output = output.view(batch_size, n_classes)
                # 1001 classes
                loss = self.loss_fn(output, y_with_bg_var)
                loss.backward()

                # Normalize and scale gradient
                normed_grad = self.step_alpha2 * torch.sign(
                    X2_var.grad.data)
                # Perturb current input image by normalized and scaled grad
                step_adv = X2_var.data + normed_grad
                # Compare with original X to keep max_epsilon limit
                total_adv = step_adv - X2
                total_adv = torch.clamp(total_adv, -self.eps2, self.eps2)
                # Apply total adv perturbation to orig images and clip
                X_adv = X2 + total_adv
                X_adv = torch.clamp(X_adv, -1.0, 1.0)
                X2_var.data = X_adv
            step_counter += 1

        # value range: (-1, 1) -> (0, 1)
        for t in X2_var.data:
            t.add_(1.0).div_(2.0)
        X2_var.data = torch.clamp(X2_var.data, 0.0, 1.0)

        # Verification
        return self.validate_and_clamp(X2_var.data, X_val)

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

    print("load models...")
    # NN Model
    m1 = densenet161(pretrained=False)
    m1 = m1.cuda()
    m1.load_state_dict(torch.load(FN_MODEL1))
    m1.eval()

    # NN2
    # m2 = resnet34(pretrained=False)
    # m2 = m2.cuda()
    # m2.load_state_dict(torch.load(FN_MODEL2))
    # m2.eval()
    m2 = fbresnet152(pretrained='imagenet')
    m2 = m2.cuda()
    m2.load_state_dict(torch.load(FN_MODEL2))
    m2.eval()

    # NN3
    m3 = Inception3(transform_input=False)
    m3 = m3.cuda()
    m3.load_state_dict(torch.load(FN_MODEL3))
    m3.eval()

    # NN Model
    m4 = inceptionresnetv2(
        num_classes=1001,
        pretrained='imagenet+background')
    m4 = m4.cuda()
    m4.load_state_dict(torch.load(FN_MODEL4))
    m4.eval()

    # NN Model
    m5 = inceptionv4(
        num_classes=1001,
        pretrained='imagenet+background')
    m5 = m5.cuda()
    m5.load_state_dict(torch.load(FN_MODEL5))
    m5.eval()

    # NN3
    m6 = inception_v3_fullconv(num_classes=1001,
                               pretrained=False)
    m6 = m6.cuda()
    m6.load_state_dict(torch.load(FN_MODEL6))
    m6.eval()

    # NN3
    m7 = inception_v3_fullconv(num_classes=1001,
                               pretrained=False)
    m7 = m7.cuda()
    m7.load_state_dict(torch.load(FN_MODEL7))
    m7.eval()

    # NN3
    m8 = inception_v3_fullconv(num_classes=1001,
                               pretrained=False)
    m8 = m8.cuda()
    m8.load_state_dict(torch.load(FN_MODEL8))
    m8.eval()

    # NN Model
    m9 = inceptionresnetv2(
        num_classes=1001,
        pretrained='imagenet+background')
    m9 = m9.cuda()
    m9.load_state_dict(torch.load(FN_MODEL9))
    m9.eval()

    # NN10
    m10 = resnext101_64x4d(
        num_classes=1000,
        pretrained='imagenet')
    m10 = m10.cuda()
    state_dict_features = torch.load(FN_MODEL10)
    state_dict_fc = collections.OrderedDict()
    state_dict_fc['weight'] = state_dict_features['10.1.weight']
    state_dict_fc['bias'] = state_dict_features['10.1.bias']
    del state_dict_features['10.1.weight']
    del state_dict_features['10.1.bias']
    m10.features.load_state_dict(state_dict_features)
    m10.fc.load_state_dict(state_dict_fc)
    m10.eval()

    # Dataset
    ds1 = Dataset(
         input_dir,
         transform=default_transform(224))
    ds2 = Dataset(
        input_dir,
        transform=default_inception_transform(299))
    ds3 = Dataset(
         input_dir,
         transform=default_transform_v2(224))
    ds_orig = Dataset(
        input_dir,
        transform=original_transform(299))

    # DataLoader
    loader1 = DataLoader(ds1, batch_size=10, shuffle=False)
    loader2 = DataLoader(ds2, batch_size=10, shuffle=False)
    loader_orig = DataLoader(ds_orig, batch_size=10, shuffle=False)

    # Attacker
    attacker = RoundRobinEnsemble(
        [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10],
        max_eps=max_epsilon,
        num_steps=10 * 2,
        n_iter=4)  # n_iter * n_models  = 9 * 2

    preds = []
    filenames = []
    loader2_iter = iter(loader2)
    loader_orig_iter = iter(loader_orig)
    for idx, (X, y) in enumerate(loader1):
        X2, y2 = loader2_iter.next()
        X_, y_ = loader_orig_iter.next()

        filenames = loader1.get_filenames(idx, len(X))
        X = X.cuda()
        X2 = X2.cuda()
        X_ = X_.cuda()
        X_adv = attacker.generate(X, X2, X_)

        for i, filename in enumerate(filenames):
            im = X_adv[i]
            im = np.clip(im * 255, 0, 255).astype(np.uint8)

            out_path = str(Path(output_name) / Path(filename))
            imsave(out_path, im, format='png')


if __name__ == '__main__':
    main()
