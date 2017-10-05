# -*- coding: utf-8 -*-
import os
import sys
import math
from pathlib import Path

import scipy.misc
import h5py
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import click

from models.slim.nets.inception_resnet_v2 import (
    inception_resnet_v2,
    inception_resnet_v2_arg_scope)


slim = tf.contrib.slim


FMT_CONV = 'InceptionResnetV2/InceptionResnetV2/{}/convolution'
FMT_RELU = 'InceptionResnetV2/InceptionResnetV2/{}/Relu'
FMT_OTHER = 'InceptionResnetV2/{}/{}'
TEST_THRESHOLD = 1e-2


def _make_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")

    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        return [
            math.floor(int(conv_shape[0])/2),
            math.floor(int(conv_shape[1])/2)
        ]
    else:
        raise RuntimeError(f"Invalid padding name: {padding_name}")


def get_store_path(outdir, name):
    return (
        Path(outdir) /
        Path('EnsAdvInceptionResnetV2') /
        Path(f'{name}.h5'))


def dump_conv2d(sess, name='Conv2d_1a_3x3', outdir='./dump'):
    conv_operation = sess.graph.get_operation_by_name(FMT_CONV.format(name))
    weights_tensor = sess.graph.get_tensor_by_name(
        FMT_OTHER.format(name, 'weights:0'))
    weights = weights_tensor.eval()

    padding = _make_padding(
        conv_operation.get_attr('padding'),
        weights_tensor.get_shape())
    strides = conv_operation.get_attr('strides')
    conv_out = sess.graph.get_operation_by_name(
        FMT_CONV.format(name)).outputs[0].eval()

    bn_beta = sess.graph.get_tensor_by_name(
        FMT_OTHER.format(name, 'BatchNorm/beta:0')).eval()
    bn_mean = sess.graph.get_tensor_by_name(
        FMT_OTHER.format(name, 'BatchNorm/moving_mean:0')).eval()
    bn_var = sess.graph.get_tensor_by_name(
        FMT_OTHER.format(name, 'BatchNorm/moving_variance:0')).eval()

    relu_out = sess.graph.get_operation_by_name(
        FMT_RELU.format(name)).outputs[0].eval()

    store_path = get_store_path(outdir, name)
    if not store_path.parent.exists():
        store_path.parent.mkdir(parents=True)

    with h5py.File(str(store_path), 'w') as h5f:
        # conv
        h5f.create_dataset("weights", data=weights)
        h5f.create_dataset("strides", data=strides)
        h5f.create_dataset("padding", data=padding)
        h5f.create_dataset("conv_out", data=conv_out)

        # batch norm
        h5f.create_dataset("beta", data=bn_beta)
        h5f.create_dataset("mean", data=bn_mean)
        h5f.create_dataset("var", data=bn_var)
        h5f.create_dataset("relu_out", data=relu_out)


def dump_conv2d_nobn(sess, name='Conv2d_1x1', outdir='./dump'):
    conv_operation = sess.graph.get_operation_by_name(FMT_CONV.format(name))
    weights_tensor = sess.graph.get_tensor_by_name(
        FMT_OTHER.format(name, 'weights:0'))
    weights = weights_tensor.eval()

    biases_tensor = sess.graph.get_tensor_by_name(
        FMT_OTHER.format(name, 'biases:0'))
    biases = biases_tensor.eval()

    padding = _make_padding(
        conv_operation.get_attr('padding'),
        weights_tensor.get_shape())
    strides = conv_operation.get_attr('strides')

    conv_out = sess.graph.get_operation_by_name(
        'InceptionResnetV2/InceptionResnetV2/' +
        name +
        '/BiasAdd').outputs[0].eval()

    store_path = get_store_path(outdir, name)
    if not store_path.parent.exists():
        store_path.parent.mkdir(parents=True)

    with h5py.File(str(store_path), 'w') as h5f:
        h5f.create_dataset("weights", data=weights)
        h5f.create_dataset("biases", data=biases)
        h5f.create_dataset("strides", data=strides)
        h5f.create_dataset("padding", data=padding)
        h5f.create_dataset("conv_out", data=conv_out)


def dump_logits(sess, outdir='./dump'):
    operation = sess.graph.get_operation_by_name(
        FMT_OTHER.format('Logits', 'Predictions'))
    weights_tensor = sess.graph.get_tensor_by_name(
        FMT_OTHER.format('Logits', 'Logits/weights:0'))
    weights = weights_tensor.eval()

    biases_tensor = sess.graph.get_tensor_by_name(
        FMT_OTHER.format('Logits', 'Logits/biases:0'))
    biases = biases_tensor.eval()

    out = operation.outputs[0].eval()

    store_path = get_store_path(outdir, 'Logits')
    if not store_path.parent.exists():
        store_path.parent.mkdir(parents=True)

    with h5py.File(str(store_path), 'w') as h5f:
        h5f.create_dataset("weights", data=weights)
        h5f.create_dataset("biases", data=biases)
        h5f.create_dataset("out", data=out)


def dump_mixed_5b(sess, name='Mixed_5b', outdir='./dump'):
    dump_conv2d(sess, name=name+'/Branch_0/Conv2d_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0b_5x5', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_2/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_2/Conv2d_0b_3x3', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_2/Conv2d_0c_3x3', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_3/Conv2d_0b_1x1', outdir=outdir)


def dump_block35(sess, name='Repeat/block35_1', outdir='./dump'):
    dump_conv2d(sess, name=name+'/Branch_0/Conv2d_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0b_3x3', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_2/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_2/Conv2d_0b_3x3', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_2/Conv2d_0c_3x3', outdir=outdir)
    dump_conv2d_nobn(sess, name=name+'/Conv2d_1x1', outdir=outdir)


def dump_mixed_6a(sess, name='Mixed_6a', outdir='./dump'):
    dump_conv2d(sess, name=name+'/Branch_0/Conv2d_1a_3x3', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0b_3x3', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_1a_3x3', outdir=outdir)


def dump_block17(sess, name='Repeat_1/block17_1', outdir='./dump'):
    dump_conv2d(sess, name=name+'/Branch_0/Conv2d_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0b_1x7', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0c_7x1', outdir=outdir)
    dump_conv2d_nobn(sess, name=name+'/Conv2d_1x1', outdir=outdir)


def dump_mixed_7a(sess, name='Mixed_7a', outdir='./dump'):
    dump_conv2d(sess, name=name+'/Branch_0/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_0/Conv2d_1a_3x3', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_1a_3x3', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_2/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_2/Conv2d_0b_3x3', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_2/Conv2d_1a_3x3', outdir=outdir)


def dump_block8(sess, name='Repeat_2/block8_1', outdir='./dump'):
    dump_conv2d(sess, name=name+'/Branch_0/Conv2d_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0b_1x3', outdir=outdir)
    dump_conv2d(sess, name=name+'/Branch_1/Conv2d_0c_3x1', outdir=outdir)
    dump_conv2d_nobn(sess, name=name+'/Conv2d_1x1', outdir=outdir)


def _assign_from_checkpoint(sess, checkpoint):
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint,
        slim.get_model_variables('InceptionResnetV2'))
    init_fn(sess)


def show_all_variables():
    for v in slim.get_model_variables():
        print(v.name, v.get_shape())


def dump_all(sess, logits, outdir):
    tf.summary.scalar('logs', logits[0][0])
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("logs", sess.graph)

    # run for comparing output values later
    out = sess.run(summary_op)
    summary_writer.add_summary(out, 0)

    dump_conv2d(sess, name='Conv2d_1a_3x3', outdir=outdir)
    dump_conv2d(sess, name='Conv2d_1a_3x3', outdir=outdir)
    dump_conv2d(sess, name='Conv2d_2a_3x3', outdir=outdir)
    dump_conv2d(sess, name='Conv2d_2b_3x3', outdir=outdir)
    dump_conv2d(sess, name='Conv2d_3b_1x1', outdir=outdir)
    dump_conv2d(sess, name='Conv2d_4a_3x3', outdir=outdir)

    # Mixed component
    dump_mixed_5b(sess, outdir=outdir)
    for i in range(1, 11):
        dump_block35(sess, name=f'Repeat/block35_{i}', outdir=outdir)

    # Mixed component
    dump_mixed_6a(sess, outdir=outdir)
    for i in range(1, 21):
        dump_block17(sess, name=f'Repeat_1/block17_{i}', outdir=outdir)

    # Mixed component
    dump_mixed_7a(sess, outdir=outdir)
    for i in range(1, 10):
        dump_block8(sess, name=f'Repeat_2/block8_{i}', outdir=outdir)

    dump_block8(sess, name='Block8', outdir=outdir)
    dump_conv2d(sess, name='Conv2d_7b_1x1', outdir=outdir)
    dump_logits(sess, outdir=outdir)


def dump_proc(checkpoint, outdir, verbose):
    with tf.Graph().as_default(),\
         slim.arg_scope(inception_resnet_v2_arg_scope()):

        inputs = np.ones((1, 299, 299, 3), dtype=np.float32)
        inputs = tf.stack(inputs)
        logits, _ = inception_resnet_v2(inputs,
                                        num_classes=1001,
                                        is_training=False)

        with tf.Session() as sess:
            _assign_from_checkpoint(sess, checkpoint)
            if verbose > 0:
                show_all_variables()

            dump_all(sess, logits, outdir)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,
                                 momentum=0,
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160,
                        kernel_size=(1, 7),
                        stride=1,
                        padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):
    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224,
                        kernel_size=(1, 3),
                        stride=1,
                        padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResnetV2(nn.Module):
    def __init__(self, num_classes=1001):
        super(InceptionResnetV2, self).__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            *[Block35(scale=0.17) for idx in range(10)])
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            *[Block17(scale=0.10) for idx in range(20)])
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            *[Block8(scale=0.20) for idx in range(9)])
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.classif = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        x = self.avgpool_1a(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return x


def inceptionresnetv2(pretrained=True):
    return InceptionResnetV2()


def load_conv2d(state, outdir, name, path):
    store_path = str(
        Path(outdir) /
        Path("EnsAdvInceptionResnetV2") /
        Path("{}.h5".format(path)))

    with h5py.File(store_path, 'r') as h5f:
        state[f'{name}.conv.weight'] = torch.from_numpy(
            h5f['weights'][()]).permute(3, 2, 0, 1)
        out_planes = state['{}.conv.weight'.format(name)].size(0)

        state[f'{name}.bn.weight'] = torch.ones(out_planes)
        state[f'{name}.bn.bias'] = torch.from_numpy(h5f['beta'][()])
        state[f'{name}.bn.running_mean'] = torch.from_numpy(h5f['mean'][()])
        state[f'{name}.bn.running_var'] = torch.from_numpy(h5f['var'][()])


def load_conv2d_nobn(state, outdir, name, path):
    store_path = str(
        Path(outdir) /
        Path("EnsAdvInceptionResnetV2") /
        Path(f"{path}.h5"))

    with h5py.File(store_path, 'r') as h5f:
        state[f'{name}.weight'] = torch.from_numpy(
            h5f['weights'][()]).permute(3, 2, 0, 1)
        state[f'{name}.bias'] = torch.from_numpy(h5f['biases'][()])


def load_mixed_5b(state, outdir, name, path):
    name_path_pairs = [
        (f'{name}.branch0',   f'{path}/Branch_0/Conv2d_1x1'),
        (f'{name}.branch1.0', f'{path}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch1.1', f'{path}/Branch_1/Conv2d_0b_5x5'),
        (f'{name}.branch2.0', f'{path}/Branch_2/Conv2d_0a_1x1'),
        (f'{name}.branch2.1', f'{path}/Branch_2/Conv2d_0b_3x3'),
        (f'{name}.branch2.2', f'{path}/Branch_2/Conv2d_0c_3x3'),
        (f'{name}.branch3.1', f'{path}/Branch_3/Conv2d_0b_1x1'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_block35(state, outdir, name, path):
    name_path_pairs = [
        (f'{name}.branch0',   f'{path}/Branch_0/Conv2d_1x1'),
        (f'{name}.branch1.0', f'{path}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch1.1', f'{path}/Branch_1/Conv2d_0b_3x3'),
        (f'{name}.branch2.0', f'{path}/Branch_2/Conv2d_0a_1x1'),
        (f'{name}.branch2.1', f'{path}/Branch_2/Conv2d_0b_3x3'),
        (f'{name}.branch2.2', f'{path}/Branch_2/Conv2d_0c_3x3'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)
    load_conv2d_nobn(state, outdir, f'{name}.conv2d', f'{path}/Conv2d_1x1')


def load_mixed_6a(state, outdir, name, path):
    name_path_pairs = [
        (f'{name}.branch0',   f'{path}/Branch_0/Conv2d_1a_3x3'),
        (f'{name}.branch1.0', f'{path}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch1.1', f'{path}/Branch_1/Conv2d_0b_3x3'),
        (f'{name}.branch1.2', f'{path}/Branch_1/Conv2d_1a_3x3'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_block17(state, outdir, name, path):
    name_path_pairs = [
        (f'{name}.branch0',   f'{path}/Branch_0/Conv2d_1x1'),
        (f'{name}.branch1.0', f'{path}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch1.1', f'{path}/Branch_1/Conv2d_0b_1x7'),
        (f'{name}.branch1.2', f'{path}/Branch_1/Conv2d_0c_7x1'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)
    load_conv2d_nobn(state, outdir, f'{name}.conv2d', f'{path}/Conv2d_1x1')


def load_mixed_7a(state, outdir, name, path):
    name_path_pairs = [
        (f'{name}.branch0.0', f'{path}/Branch_0/Conv2d_0a_1x1'),
        (f'{name}.branch0.1', f'{path}/Branch_0/Conv2d_1a_3x3'),
        (f'{name}.branch1.0', f'{path}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch1.1', f'{path}/Branch_1/Conv2d_1a_3x3'),
        (f'{name}.branch2.0', f'{path}/Branch_2/Conv2d_0a_1x1'),
        (f'{name}.branch2.1', f'{path}/Branch_2/Conv2d_0b_3x3'),
        (f'{name}.branch2.2', f'{path}/Branch_2/Conv2d_1a_3x3'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_block8(state, outdir, name, path):
    name_path_pairs = [
        (f'{name}.branch0',   f'{path}/Branch_0/Conv2d_1x1'),
        (f'{name}.branch1.0', f'{path}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch1.1', f'{path}/Branch_1/Conv2d_0b_1x3'),
        (f'{name}.branch1.2', f'{path}/Branch_1/Conv2d_0c_3x1'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)
    load_conv2d_nobn(state, outdir, f'{name}.conv2d', f'{path}/Conv2d_1x1')


def load_linear(state, outdir, name, path):
    store_path = str(
        Path(outdir) /
        Path("EnsAdvInceptionResnetV2") /
        Path(f"{path}.h5"))

    with h5py.File(store_path, 'r') as h5f:
        state[f'{name}.weight'] = torch.from_numpy(h5f['weights'][()]).t()
        state[f'{name}.bias'] = torch.from_numpy(h5f['biases'][()])


def load_state_dict_from_h5py(outdir):
    state = {}
    load_conv2d(state, outdir, 'conv2d_1a', 'Conv2d_1a_3x3')
    load_conv2d(state, outdir, 'conv2d_2a', 'Conv2d_2a_3x3')
    load_conv2d(state, outdir, 'conv2d_2b', 'Conv2d_2b_3x3')
    load_conv2d(state, outdir, 'conv2d_3b', 'Conv2d_3b_1x1')
    load_conv2d(state, outdir, 'conv2d_4a', 'Conv2d_4a_3x3')

    load_mixed_5b(state, outdir, 'mixed_5b', 'Mixed_5b')

    for i in range(10):
        load_block35(state, outdir, f'repeat.{i}', f'Repeat/block35_{i+1}')

    load_mixed_6a(state, outdir, 'mixed_6a', 'Mixed_6a')

    for i in range(20):
        load_block17(state, outdir, f'repeat_1.{i}', f'Repeat_1/block17_{i+1}')

    load_mixed_7a(state, outdir, 'mixed_7a', 'Mixed_7a')

    for i in range(9):
        load_block8(state, outdir, f'repeat_2.{i}', f'Repeat_2/block8_{i+1}')

    load_block8(state, outdir, 'block8', 'Block8')
    load_conv2d(state, outdir, 'conv2d_7b', 'Conv2d_7b_1x1')
    load_linear(state, outdir, 'classif', 'Logits')

    return state


def load_proc(outdir, export_path, verbose):
    model = InceptionResnetV2()
    state_dict = load_state_dict_from_h5py(outdir)
    model.load_state_dict(state_dict)
    model.eval()
    torch.save(state_dict, export_path)
    return model


def test_conv2d(outdir, module, path):
    store_path = str(
        Path(outdir) /
        Path("EnsAdvInceptionResnetV2") /
        Path(f"{path}.h5"))

    with h5py.File(store_path, 'r') as h5f:
        output_tf_conv = torch.from_numpy(h5f['conv_out'][()])
        output_tf_conv.transpose_(1, 3)
        output_tf_conv.transpose_(2, 3)
        output_tf_relu = torch.from_numpy(h5f['relu_out'][()])
        output_tf_relu.transpose_(1, 3)
        output_tf_relu.transpose_(2, 3)

    def test_dist_conv(self, input, output):
        dist = torch.dist(output.data, output_tf_conv)
        assert dist < TEST_THRESHOLD

    def test_dist_relu(self, input, output):
        dist = torch.dist(output.data, output_tf_relu)
        assert dist < TEST_THRESHOLD

    module.conv.register_forward_hook(test_dist_conv)
    module.relu.register_forward_hook(test_dist_relu)


def test_conv2d_nobn(outdir, module, path):
    store_path = str(
        Path(outdir) /
        Path("EnsAdvInceptionResnetV2") /
        Path(f"{path}.h5"))
    with h5py.File(store_path, 'r') as h5f:
        output_tf = torch.from_numpy(h5f['conv_out'][()])
        output_tf.transpose_(1, 3)
        output_tf.transpose_(2, 3)

    def test_dist(self, input, output):
        dist = torch.dist(output.data, output_tf)
        assert dist < TEST_THRESHOLD

    module.register_forward_hook(test_dist)


def _register_forward_hook(outdir, model):
    test_conv2d(outdir, model.conv2d_1a, 'Conv2d_1a_3x3')
    test_conv2d(outdir, model.conv2d_2a, 'Conv2d_2a_3x3')
    test_conv2d(outdir, model.conv2d_2b, 'Conv2d_2b_3x3')
    test_conv2d(outdir, model.conv2d_3b, 'Conv2d_3b_1x1')
    test_conv2d(outdir, model.conv2d_4a, 'Conv2d_4a_3x3')

    test_mixed_5b(outdir, model.mixed_5b, 'Mixed_5b')
    for i in range(len(model.repeat._modules)):
        test_block35(outdir, model.repeat[i], f'Repeat/block35_{i+1}')

    test_mixed_6a(outdir, model.mixed_6a, 'Mixed_6a')

    for i in range(len(model.repeat_1._modules)):
        test_block17(outdir, model.repeat_1[i], f'Repeat_1/block17_{i+1}')

    test_mixed_7a(outdir, model.mixed_7a, 'Mixed_7a')

    for i in range(len(model.repeat_2._modules)):
        test_block8(outdir, model.repeat_2[i], f'Repeat_2/block8_{i+1}')

    test_block8(outdir, model.block8, 'Block8')
    test_conv2d(outdir, model.conv2d_7b, 'Conv2d_7b_1x1')


def test_mixed_5b(outdir, module, name):
    test_conv2d(outdir, module.branch0,    f'{name}/Branch_0/Conv2d_1x1')
    test_conv2d(outdir, module.branch1[0], f'{name}/Branch_1/Conv2d_0a_1x1')
    test_conv2d(outdir, module.branch1[1], f'{name}/Branch_1/Conv2d_0b_5x5')
    test_conv2d(outdir, module.branch2[0], f'{name}/Branch_2/Conv2d_0a_1x1')
    test_conv2d(outdir, module.branch2[1], f'{name}/Branch_2/Conv2d_0b_3x3')
    test_conv2d(outdir, module.branch2[2], f'{name}/Branch_2/Conv2d_0c_3x3')
    test_conv2d(outdir, module.branch3[1], f'{name}/Branch_3/Conv2d_0b_1x1')


def test_block35(outdir, module, name):
    test_conv2d(outdir, module.branch0, f'{name}/Branch_0/Conv2d_1x1')
    test_conv2d(outdir, module.branch1[0], f'{name}/Branch_1/Conv2d_0a_1x1')
    test_conv2d(outdir, module.branch1[1], f'{name}/Branch_1/Conv2d_0b_3x3')
    test_conv2d(outdir, module.branch2[0], f'{name}/Branch_2/Conv2d_0a_1x1')
    test_conv2d(outdir, module.branch2[1], f'{name}/Branch_2/Conv2d_0b_3x3')
    test_conv2d(outdir, module.branch2[2], f'{name}/Branch_2/Conv2d_0c_3x3')
    test_conv2d_nobn(outdir, module.conv2d, f'{name}/Conv2d_1x1')


def test_mixed_6a(outdir, module, name):
    test_conv2d(outdir, module.branch0, f'{name}/Branch_0/Conv2d_1a_3x3')
    test_conv2d(outdir, module.branch1[0], f'{name}/Branch_1/Conv2d_0a_1x1')
    test_conv2d(outdir, module.branch1[1], f'{name}/Branch_1/Conv2d_0b_3x3')
    test_conv2d(outdir, module.branch1[2], f'{name}/Branch_1/Conv2d_1a_3x3')


def test_block17(outdir, module, name):
    test_conv2d(outdir, module.branch0, f'{name}/Branch_0/Conv2d_1x1')
    test_conv2d(outdir, module.branch1[0], f'{name}/Branch_1/Conv2d_0a_1x1')
    test_conv2d(outdir, module.branch1[1], f'{name}/Branch_1/Conv2d_0b_1x7')
    test_conv2d(outdir, module.branch1[2], f'{name}/Branch_1/Conv2d_0c_7x1')
    test_conv2d_nobn(outdir, module.conv2d, f'{name}/Conv2d_1x1')


def test_mixed_7a(outdir, module, name):
    test_conv2d(outdir, module.branch0[0], f'{name}/Branch_0/Conv2d_0a_1x1')
    test_conv2d(outdir, module.branch0[1], f'{name}/Branch_0/Conv2d_1a_3x3')
    test_conv2d(outdir, module.branch1[0], f'{name}/Branch_1/Conv2d_0a_1x1')
    test_conv2d(outdir, module.branch1[1], f'{name}/Branch_1/Conv2d_1a_3x3')
    test_conv2d(outdir, module.branch2[0], f'{name}/Branch_2/Conv2d_0a_1x1')
    test_conv2d(outdir, module.branch2[1], f'{name}/Branch_2/Conv2d_0b_3x3')
    test_conv2d(outdir, module.branch2[2], f'{name}/Branch_2/Conv2d_1a_3x3')


def test_block8(outdir, module, name):
    test_conv2d(outdir, module.branch0, f'{name}/Branch_0/Conv2d_1x1')
    test_conv2d(outdir, module.branch1[0], f'{name}/Branch_1/Conv2d_0a_1x1')
    test_conv2d(outdir, module.branch1[1], f'{name}/Branch_1/Conv2d_0b_1x3')
    test_conv2d(outdir, module.branch1[2], f'{name}/Branch_1/Conv2d_0c_3x1')
    test_conv2d_nobn(outdir, module.conv2d, f'{name}/Conv2d_1x1')


def run_test(outdir, model):
    _register_forward_hook(outdir, model)

    inputs = np.ones((1, 299, 299, 3), dtype=np.float32)
    inputs = torch.from_numpy(inputs)
    inputs = inputs * 2 - 1.0
    inputs = inputs.permute(0, 3, 1, 2)

    store_path = str(
        Path(outdir) /
        Path("EnsAdvInceptionResnetV2") /
        Path("Logits.h5"))
    outputs = model.forward(torch.autograd.Variable(inputs))
    with h5py.File(store_path, 'r') as h5f:
        outputs_tf = torch.from_numpy(h5f['out'][()])
    outputs = torch.nn.functional.softmax(outputs)

    dist = torch.dist(outputs.data, outputs_tf)
    assert dist < TEST_THRESHOLD
    print("OK")


@click.command()
@click.option('--checkpoint',
              help='checkpoint file',
              default='./ens_adv_inception_resnet_v2.ckpt')
@click.option('--outdir',
              help='output directory',
              default='./dump')
@click.option('--export-path',
              default='../working_files/ensadv_inceptionresnetv2_state.pth')
@click.option('-v', '--verbose', count=True)
def main(checkpoint, outdir, export_path, verbose):
    dump_proc(checkpoint, outdir, verbose)
    model = load_proc(outdir, export_path, verbose)
    if verbose > 0:
        run_test(outdir, model)


if __name__ == '__main__':
    main()
