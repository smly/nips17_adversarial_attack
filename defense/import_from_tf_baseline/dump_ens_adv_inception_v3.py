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

from models.slim.nets.inception_v3 import (
    inception_v3,
    inception_v3_arg_scope)
import inception_v3_fullconv


slim = tf.contrib.slim


FMT_CONV = 'InceptionV3/InceptionV3/{}/convolution'
FMT_RELU = 'InceptionV3/InceptionV3/{}/Relu'
FMT_OTHER = 'InceptionV3/{}/{}'
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
        Path('dump') /
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


def dump_logits(sess, name='Logits/Conv2d_1c_1x1', outdir='./dump'):
    conv_operation = sess.graph.get_operation_by_name(
        f'InceptionV3/{name}/convolution')
    weights_tensor = sess.graph.get_tensor_by_name(
        f'InceptionV3/{name}/weights:0')
    weights = weights_tensor.eval()
    biases_tensor = sess.graph.get_tensor_by_name(
        f'InceptionV3/{name}/biases:0')
    biases = biases_tensor.eval()

    padding = _make_padding(
        conv_operation.get_attr('padding'),
        weights_tensor.get_shape())
    strides = conv_operation.get_attr('strides')

    conv_out = sess.graph.get_operation_by_name(
        f'InceptionV3/{name}/BiasAdd').outputs[0].eval()

    store_path = get_store_path(outdir, name)
    if not store_path.parent.exists():
        store_path.parent.mkdir(parents=True)

    with h5py.File(str(store_path), 'w') as h5f:
        h5f.create_dataset("weights", data=weights)
        h5f.create_dataset("biases", data=biases)
        h5f.create_dataset("strides", data=strides)
        h5f.create_dataset("padding", data=padding)
        h5f.create_dataset("conv_out", data=conv_out)


def dump_conv2d_logit(sess, name='Conv2d_1a_3x3', outdir='./dump'):
    conv_operation = sess.graph.get_operation_by_name(
        'InceptionV3/{}/convolution'.format(name))
    weights_tensor = sess.graph.get_tensor_by_name(
        'InceptionV3/{}/weights:0'.format(name))
    weights = weights_tensor.eval()

    padding = _make_padding(
        conv_operation.get_attr('padding'),
        weights.shape)
    strides = conv_operation.get_attr('strides')

    conv_out = sess.graph.get_operation_by_name(
        'InceptionV3/{}/convolution'.format(name)
    ).outputs[0].eval()
    beta = sess.graph.get_tensor_by_name(
        'InceptionV3/{}/BatchNorm/beta:0'.format(name)).eval()
    mean = sess.graph.get_tensor_by_name(
        'InceptionV3/{}/BatchNorm/moving_mean:0'.format(name)).eval()
    var = sess.graph.get_tensor_by_name(
        'InceptionV3/{}/BatchNorm/moving_variance:0'.format(name)).eval()
    relu_out = sess.graph.get_operation_by_name(
        'InceptionV3/{}/Relu'.format(name)).outputs[0].eval()

    store_path = get_store_path(outdir, name)
    if not store_path.parent.exists():
        store_path.parent.mkdir(parents=True)

    with h5py.File(str(store_path), 'w') as h5f:
        h5f.create_dataset("weights", data=weights)
        h5f.create_dataset("strides", data=strides)
        h5f.create_dataset("padding", data=padding)
        h5f.create_dataset("conv_out", data=conv_out)
        # batch norm
        h5f.create_dataset("beta", data=beta)
        h5f.create_dataset("mean", data=mean)
        h5f.create_dataset("var", data=var)
        h5f.create_dataset("relu_out", data=relu_out)


def dump_mixed_5b(sess, name='Mixed_5b', outdir='./dump'):
    dump_conv2d(sess, name=f'{name}/Branch_0/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_1/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_1/Conv2d_0b_5x5', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_2/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_2/Conv2d_0b_3x3', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_2/Conv2d_0c_3x3', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_3/Conv2d_0b_1x1', outdir=outdir)


def dump_mixed_5c(sess, name='Mixed_5c', outdir='./dump'):
    dump_conv2d(sess, name=f'{name}/Branch_0/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_1/Conv2d_0b_1x1', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_1/Conv_1_0c_5x5', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_2/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_2/Conv2d_0b_3x3', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_2/Conv2d_0c_3x3', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_3/Conv2d_0b_1x1', outdir=outdir)


def dump_mixed_5d(sess, name='Mixed_5d', outdir='./dump'):
    dump_conv2d(sess, name=f'{name}/Branch_0/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_1/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_1/Conv2d_0b_5x5', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_2/Conv2d_0a_1x1', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_2/Conv2d_0b_3x3', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_2/Conv2d_0c_3x3', outdir=outdir)
    dump_conv2d(sess, name=f'{name}/Branch_3/Conv2d_0b_1x1', outdir=outdir)


def dump_mixed_6a(sess, name='Mixed_6a', outdir='./dump'):
    dump_conv2d(sess, f"{name}/Branch_0/Conv2d_1a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0b_3x3", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_1a_1x1", outdir=outdir)


def dump_mixed_6b(sess, name='Mixed_6b', outdir='./dump'):
    dump_conv2d(sess, f"{name}/Branch_0/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0b_1x7", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0c_7x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0b_7x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0c_1x7", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0d_7x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0e_1x7", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_3/Conv2d_0b_1x1", outdir=outdir)


def dump_mixed_6c(sess, name='Mixed_6c', outdir='./dump'):
    dump_mixed_6b(sess, name=name, outdir=outdir)


def dump_mixed_6d(sess, name='Mixed_6d', outdir='./dump'):
    dump_mixed_6b(sess, name=name, outdir=outdir)


def dump_mixed_6e(sess, name='Mixed_6e', outdir='./dump'):
    dump_mixed_6b(sess, name=name, outdir=outdir)


def dump_mixed_7a(sess, name='Mixed_7a', outdir='./dump'):
    dump_conv2d(sess, f"{name}/Branch_0/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_0/Conv2d_1a_3x3", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0b_1x7", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0c_7x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_1a_3x3", outdir=outdir)


def dump_mixed_7b(sess, name='Mixed_7b', outdir='./dump'):
    dump_conv2d(sess, f"{name}/Branch_0/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0b_1x3", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0b_3x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0b_3x3", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0c_1x3", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0d_3x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_3/Conv2d_0b_1x1", outdir=outdir)


def dump_mixed_7c(sess, name='Mixed_7c', outdir='./dump'):
    dump_conv2d(sess, f"{name}/Branch_0/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0b_1x3", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_1/Conv2d_0c_3x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0a_1x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0b_3x3", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0c_1x3", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_2/Conv2d_0d_3x1", outdir=outdir)
    dump_conv2d(sess, f"{name}/Branch_3/Conv2d_0b_1x1", outdir=outdir)


def dump_aux_logits(sess, outdir='./dump'):
    dump_conv2d_logit(sess, "AuxLogits/Conv2d_1b_1x1", outdir=outdir)
    dump_conv2d_logit(sess, "AuxLogits/Conv2d_2a_5x5", outdir=outdir)

    weights_tensor = sess.graph.get_tensor_by_name(
        'InceptionV3/AuxLogits/Conv2d_2b_1x1/weights:0')
    weights = weights_tensor.eval()
    biases_tensor = sess.graph.get_tensor_by_name(
        'InceptionV3/AuxLogits/Conv2d_2b_1x1/biases:0')
    biases = biases_tensor.eval()

    store_path = get_store_path(outdir, 'AuxLogits')
    if not store_path.parent.exists():
        store_path.parent.mkdir(parents=True)

    with h5py.File(str(store_path), 'w') as h5f:
        h5f.create_dataset("weights", data=weights)
        h5f.create_dataset("biases", data=biases)


def _assign_from_checkpoint(sess, checkpoint):
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint,
        slim.get_model_variables('InceptionV3'))
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

    dump_logits(sess, outdir=outdir)
    dump_conv2d(sess, name='Conv2d_1a_3x3', outdir=outdir)
    dump_conv2d(sess, name='Conv2d_2a_3x3', outdir=outdir)
    dump_conv2d(sess, name='Conv2d_2b_3x3', outdir=outdir)
    dump_conv2d(sess, name='Conv2d_3b_1x1', outdir=outdir)
    dump_conv2d(sess, name='Conv2d_4a_3x3', outdir=outdir)

    dump_mixed_5b(sess, outdir=outdir)
    dump_mixed_5c(sess, outdir=outdir)
    dump_mixed_5d(sess, outdir=outdir)
    dump_mixed_6a(sess, outdir=outdir)

    dump_mixed_6b(sess, outdir=outdir)
    dump_mixed_6c(sess, outdir=outdir)
    dump_mixed_6d(sess, outdir=outdir)
    dump_mixed_6e(sess, outdir=outdir)

    dump_mixed_7a(sess, outdir=outdir)
    dump_mixed_7b(sess, outdir=outdir)
    dump_mixed_7c(sess, outdir=outdir)

    dump_aux_logits(sess, outdir=outdir)


def dump_proc(checkpoint, outdir, verbose):
    with tf.Graph().as_default(),\
         slim.arg_scope(inception_v3_arg_scope()):

        img = scipy.misc.imread('./000b7d55b6184b08.png')
        inputs = np.ones((1, 299, 299, 3), dtype=np.float32)
        inputs[0] = (img / 255.0) * 2 - 1.0
        inputs = tf.stack(inputs)
        logits, _ = inception_v3(inputs,
                                 num_classes=1001,
                                 is_training=False)

        with tf.Session() as sess:
            _assign_from_checkpoint(sess, checkpoint)
            if verbose > 0:
                show_all_variables()

            dump_all(sess, logits, outdir)


def load_conv2d(state, outdir, name, path):
    store_path = str(
        Path(outdir) /
        Path("dump") /
        Path("{}.h5".format(path)))

    with h5py.File(store_path, 'r') as h5f:
        state[f'{name}.conv.weight'] = torch.from_numpy(
            h5f['weights'][()]).permute(3, 2, 0, 1)
        out_planes = state['{}.conv.weight'.format(name)].size(0)

        state[f'{name}.bn.weight'] = torch.ones(out_planes)
        state[f'{name}.bn.bias'] = torch.from_numpy(h5f['beta'][()])
        state[f'{name}.bn.running_mean'] = torch.from_numpy(h5f['mean'][()])
        state[f'{name}.bn.running_var'] = torch.from_numpy(h5f['var'][()])


def load_aux_logits(state, outdir):
    load_conv2d(state, outdir, 'AuxLogits.conv0', 'AuxLogits/Conv2d_1b_1x1')
    load_conv2d(state, outdir, 'AuxLogits.conv1', 'AuxLogits/Conv2d_2a_5x5')

    store_path = str(
        Path(outdir) /
        Path("dump") /
        Path("AuxLogits.h5"))

    with h5py.File(store_path, 'r') as h5f:
        state['AuxLogits.fc.bias'] = torch.from_numpy(h5f['biases'][()])
        state['AuxLogits.fc.weight'] = torch.from_numpy(h5f['weights'][()])

    store_path = str(
        Path(outdir) /
        Path("dump") /
        Path("Logits/Conv2d_1c_1x1.h5"))

    with h5py.File(store_path, 'r') as h5f:
        state["fc.weight"] = torch.from_numpy(
            h5f['weights'][()]).permute(3, 2, 0, 1)
        state["fc.bias"] = torch.from_numpy(h5f['biases'][()])


def load_mixed_5b(state, outdir, name):
    name_path_pairs = [
        # Branch 0
        (f'{name}.branch1x1',   f'{name}/Branch_0/Conv2d_0a_1x1'),
        # Branch 1
        (f'{name}.branch5x5_1', f'{name}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch5x5_2', f'{name}/Branch_1/Conv2d_0b_5x5'),
        # Branch 2
        (f'{name}.branch3x3dbl_1', f'{name}/Branch_2/Conv2d_0a_1x1'),
        (f'{name}.branch3x3dbl_2', f'{name}/Branch_2/Conv2d_0b_3x3'),
        (f'{name}.branch3x3dbl_3', f'{name}/Branch_2/Conv2d_0c_3x3'),
        # Branch 3
        (f'{name}.branch_pool', f'{name}/Branch_3/Conv2d_0b_1x1'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_mixed_5c(state, outdir, name):
    name_path_pairs = [
        (f'{name}.branch1x1',   f'{name}/Branch_0/Conv2d_0a_1x1'),
        (f'{name}.branch5x5_1', f'{name}/Branch_1/Conv2d_0b_1x1'),
        (f'{name}.branch5x5_2', f'{name}/Branch_1/Conv_1_0c_5x5'),
        (f'{name}.branch3x3dbl_1', f'{name}/Branch_2/Conv2d_0a_1x1'),
        (f'{name}.branch3x3dbl_2', f'{name}/Branch_2/Conv2d_0b_3x3'),
        (f'{name}.branch3x3dbl_3', f'{name}/Branch_2/Conv2d_0c_3x3'),
        (f'{name}.branch_pool', f'{name}/Branch_3/Conv2d_0b_1x1'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_mixed_5d(state, outdir, name):
    name_path_pairs = [
        (f'{name}.branch1x1',   f'{name}/Branch_0/Conv2d_0a_1x1'),
        (f'{name}.branch5x5_1', f'{name}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch5x5_2', f'{name}/Branch_1/Conv2d_0b_5x5'),
        (f'{name}.branch3x3dbl_1', f'{name}/Branch_2/Conv2d_0a_1x1'),
        (f'{name}.branch3x3dbl_2', f'{name}/Branch_2/Conv2d_0b_3x3'),
        (f'{name}.branch3x3dbl_3', f'{name}/Branch_2/Conv2d_0c_3x3'),
        (f'{name}.branch_pool', f'{name}/Branch_3/Conv2d_0b_1x1'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_mixed_6a(state, outdir, name):
    name_path_pairs = [
        (f'{name}.branch3x3',   f'{name}/Branch_0/Conv2d_1a_1x1'),
        (f'{name}.branch3x3dbl_1', f'{name}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch3x3dbl_2', f'{name}/Branch_1/Conv2d_0b_3x3'),
        (f'{name}.branch3x3dbl_3', f'{name}/Branch_1/Conv2d_1a_1x1'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_mixed_6b(state, outdir, name):
    name_path_pairs = [
        (f'{name}.branch1x1',   f'{name}/Branch_0/Conv2d_0a_1x1'),

        (f'{name}.branch7x7_1', f'{name}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch7x7_2', f'{name}/Branch_1/Conv2d_0b_1x7'),
        (f'{name}.branch7x7_3', f'{name}/Branch_1/Conv2d_0c_7x1'),

        (f'{name}.branch7x7dbl_1', f'{name}/Branch_2/Conv2d_0a_1x1'),
        (f'{name}.branch7x7dbl_2', f'{name}/Branch_2/Conv2d_0b_7x1'),
        (f'{name}.branch7x7dbl_3', f'{name}/Branch_2/Conv2d_0c_1x7'),
        (f'{name}.branch7x7dbl_4', f'{name}/Branch_2/Conv2d_0d_7x1'),
        (f'{name}.branch7x7dbl_5', f'{name}/Branch_2/Conv2d_0e_1x7'),

        (f'{name}.branch_pool', f'{name}/Branch_3/Conv2d_0b_1x1'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_mixed_6c(state, outdir, name):
    load_mixed_6b(state, outdir, name)


def load_mixed_6d(state, outdir, name):
    load_mixed_6b(state, outdir, name)


def load_mixed_6e(state, outdir, name):
    load_mixed_6b(state, outdir, name)


def load_mixed_7a(state, outdir, name):
    name_path_pairs = [
        (f'{name}.branch3x3_1', f'{name}/Branch_0/Conv2d_0a_1x1'),
        (f'{name}.branch3x3_2', f'{name}/Branch_0/Conv2d_1a_3x3'),

        (f'{name}.branch7x7x3_1', f'{name}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch7x7x3_2', f'{name}/Branch_1/Conv2d_0b_1x7'),
        (f'{name}.branch7x7x3_3', f'{name}/Branch_1/Conv2d_0c_7x1'),
        (f'{name}.branch7x7x3_4', f'{name}/Branch_1/Conv2d_1a_3x3'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_mixed_7b(state, outdir, name):
    name_path_pairs = [
        (f'{name}.branch1x1', f'{name}/Branch_0/Conv2d_0a_1x1'),
        (f'{name}.branch3x3_1', f'{name}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch3x3_2a', f'{name}/Branch_1/Conv2d_0b_1x3'),
        (f'{name}.branch3x3_2b', f'{name}/Branch_1/Conv2d_0b_3x1'),

        (f'{name}.branch3x3dbl_1', f'{name}/Branch_2/Conv2d_0a_1x1'),
        (f'{name}.branch3x3dbl_2', f'{name}/Branch_2/Conv2d_0b_3x3'),
        (f'{name}.branch3x3dbl_3a', f'{name}/Branch_2/Conv2d_0c_1x3'),
        (f'{name}.branch3x3dbl_3b', f'{name}/Branch_2/Conv2d_0d_3x1'),

        (f'{name}.branch_pool', f'{name}/Branch_3/Conv2d_0b_1x1'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_mixed_7c(state, outdir, name):
    name_path_pairs = [
        (f'{name}.branch1x1', f'{name}/Branch_0/Conv2d_0a_1x1'),

        (f'{name}.branch3x3_1', f'{name}/Branch_1/Conv2d_0a_1x1'),
        (f'{name}.branch3x3_2a', f'{name}/Branch_1/Conv2d_0b_1x3'),
        (f'{name}.branch3x3_2b', f'{name}/Branch_1/Conv2d_0c_3x1'),

        (f'{name}.branch3x3dbl_1', f'{name}/Branch_2/Conv2d_0a_1x1'),
        (f'{name}.branch3x3dbl_2', f'{name}/Branch_2/Conv2d_0b_3x3'),
        (f'{name}.branch3x3dbl_3a', f'{name}/Branch_2/Conv2d_0c_1x3'),
        (f'{name}.branch3x3dbl_3b', f'{name}/Branch_2/Conv2d_0d_3x1'),

        (f'{name}.branch_pool', f'{name}/Branch_3/Conv2d_0b_1x1'),
    ]
    for name_, path_ in name_path_pairs:
        load_conv2d(state, outdir, name_, path_)


def load_state_dict_from_h5py(outdir):
    state = {}
    load_conv2d(state, outdir, 'Conv2d_1a_3x3', 'Conv2d_1a_3x3')
    load_conv2d(state, outdir, 'Conv2d_2a_3x3', 'Conv2d_2a_3x3')
    load_conv2d(state, outdir, 'Conv2d_2b_3x3', 'Conv2d_2b_3x3')
    load_conv2d(state, outdir, 'Conv2d_3b_1x1', 'Conv2d_3b_1x1')
    load_conv2d(state, outdir, 'Conv2d_4a_3x3', 'Conv2d_4a_3x3')

    load_mixed_5b(state, outdir, 'Mixed_5b')
    load_mixed_5c(state, outdir, 'Mixed_5c')
    load_mixed_5d(state, outdir, 'Mixed_5d')

    load_mixed_6a(state, outdir, 'Mixed_6a')
    load_mixed_6b(state, outdir, 'Mixed_6b')
    load_mixed_6c(state, outdir, 'Mixed_6c')
    load_mixed_6d(state, outdir, 'Mixed_6d')
    load_mixed_6e(state, outdir, 'Mixed_6e')

    load_mixed_7a(state, outdir, 'Mixed_7a')
    load_mixed_7b(state, outdir, 'Mixed_7b')
    load_mixed_7c(state, outdir, 'Mixed_7c')

    load_aux_logits(state, outdir)

    return state


def load_proc(outdir, export_path, verbose):
    model = inception_v3_fullconv.inception_v3(
        num_classes=1001,
        pretrained=False)
    state_dict = load_state_dict_from_h5py(outdir)
    model.load_state_dict(state_dict)
    model.eval()
    torch.save(state_dict, export_path)
    return model


def test_conv2d(outdir, module, path):
    store_path = str(
        Path(outdir) /
        Path("dump") /
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
        Path("dump") /
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
    test_conv2d(outdir, model.Conv2d_1a_3x3, 'Conv2d_1a_3x3')
    test_conv2d(outdir, model.Conv2d_2a_3x3, 'Conv2d_2a_3x3')
    test_conv2d(outdir, model.Conv2d_2b_3x3, 'Conv2d_2b_3x3')
    test_conv2d(outdir, model.Conv2d_3b_1x1, 'Conv2d_3b_1x1')
    test_conv2d(outdir, model.Conv2d_4a_3x3, 'Conv2d_4a_3x3')

    test_conv2d(outdir, model.Mixed_5b.branch1x1,
                'Mixed_5b/Branch_0/Conv2d_0a_1x1')
    test_conv2d(outdir, model.Mixed_5b.branch5x5_1,
                'Mixed_5b/Branch_1/Conv2d_0a_1x1')
    test_conv2d(outdir, model.Mixed_5b.branch5x5_2,
                'Mixed_5b/Branch_1/Conv2d_0b_5x5')
    test_conv2d(outdir, model.Mixed_5b.branch3x3dbl_1,
                'Mixed_5b/Branch_2/Conv2d_0a_1x1')
    test_conv2d(outdir, model.Mixed_5b.branch3x3dbl_2,
                'Mixed_5b/Branch_2/Conv2d_0b_3x3')
    test_conv2d(outdir, model.Mixed_5b.branch3x3dbl_3,
                'Mixed_5b/Branch_2/Conv2d_0c_3x3')
    test_conv2d(outdir, model.Mixed_5b.branch_pool,
                'Mixed_5b/Branch_3/Conv2d_0b_1x1')

    test_conv2d(outdir, model.Mixed_5c.branch1x1,
                'Mixed_5c/Branch_0/Conv2d_0a_1x1')
    test_conv2d(outdir, model.Mixed_5c.branch5x5_1,
                'Mixed_5c/Branch_1/Conv2d_0b_1x1')
    test_conv2d(outdir, model.Mixed_5c.branch5x5_2,
                'Mixed_5c/Branch_1/Conv_1_0c_5x5')
    test_conv2d(outdir, model.Mixed_5c.branch3x3dbl_1,
                'Mixed_5c/Branch_2/Conv2d_0a_1x1')
    test_conv2d(outdir, model.Mixed_5c.branch3x3dbl_2,
                'Mixed_5c/Branch_2/Conv2d_0b_3x3')
    test_conv2d(outdir, model.Mixed_5c.branch3x3dbl_3,
                'Mixed_5c/Branch_2/Conv2d_0c_3x3')
    test_conv2d(outdir, model.Mixed_5c.branch_pool,
                'Mixed_5c/Branch_3/Conv2d_0b_1x1')

    test_conv2d(outdir, model.Mixed_6b.branch7x7_1,
                'Mixed_6b/Branch_1/Conv2d_0a_1x1')
    test_conv2d(outdir, model.Mixed_6b.branch7x7_2,
                'Mixed_6b/Branch_1/Conv2d_0b_1x7')
    test_conv2d(outdir, model.Mixed_6b.branch7x7_3,
                'Mixed_6b/Branch_1/Conv2d_0c_7x1')

    # 7a
    test_conv2d(outdir, model.Mixed_7a.branch3x3_1,
                'Mixed_7a/Branch_0/Conv2d_0a_1x1')
    test_conv2d(outdir, model.Mixed_7a.branch3x3_2,
                'Mixed_7a/Branch_0/Conv2d_1a_3x3')

    # 7b
    test_conv2d(outdir, model.Mixed_7b.branch3x3_1,
                'Mixed_7b/Branch_1/Conv2d_0a_1x1')
    test_conv2d(outdir, model.Mixed_7b.branch3x3_2a,
                'Mixed_7b/Branch_1/Conv2d_0b_1x3')
    test_conv2d(outdir, model.Mixed_7b.branch3x3_2b,
                'Mixed_7b/Branch_1/Conv2d_0b_3x1')

    # 7c
    test_conv2d(outdir, model.Mixed_7c.branch3x3_1,
                'Mixed_7c/Branch_1/Conv2d_0a_1x1')
    test_conv2d(outdir, model.Mixed_7c.branch3x3_2a,
                'Mixed_7c/Branch_1/Conv2d_0b_1x3')
    test_conv2d(outdir, model.Mixed_7c.branch3x3_2b,
                'Mixed_7c/Branch_1/Conv2d_0c_3x1')

    test_conv2d_nobn(outdir, model.fc, 'Logits/Conv2d_1c_1x1')


def run_test(outdir, model):
    _register_forward_hook(outdir, model)

    img = scipy.misc.imread('./000b7d55b6184b08.png')
    inputs = np.ones((1, 299, 299, 3), dtype=np.float32)
    inputs[0] = img.astype(np.float32) / 255.0
    inputs = torch.from_numpy(inputs)
    inputs = inputs * 2 - 1.0
    inputs = inputs.permute(0, 3, 1, 2)
    outputs = model.forward(torch.autograd.Variable(inputs))
    print("OK")


@click.command()
@click.option('--checkpoint',
              help='checkpoint file',
              default='./ens3_adv_inception_v3.ckpt')
@click.option('--outdir',
              help='output directory',
              default='./dump')
@click.option('--export-path',
              default='../working_files/ens3incepv3_fullconv_state.pth')
@click.option('-v', '--verbose', count=True)
def main(checkpoint, outdir, export_path, verbose):
    dump_proc(checkpoint, outdir, verbose)
    model = load_proc(outdir, export_path, verbose)
    if verbose > 0:
        run_test(outdir, model)


if __name__ == '__main__':
    main()
