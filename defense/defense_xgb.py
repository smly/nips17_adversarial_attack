# -*- coding: utf-8 -*-
import sys
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision.models.densenet import densenet121
import torch.nn.functional as F
import torch.legacy.nn as legacy_nn
import tables as tbl
import xgboost as xgb

from models_drn import drn_c_58, drn_d_105
from models_dpn import dpn131, dpn107
from models_incepv3 import inception_v3 as inception_v3_fullconv
from models_incepresv2 import inceptionresnetv2
from models_inceptionv4 import inceptionv4
import xgb_feat
from dataset import (
    Dataset, DataLoader,
    default_inception_transform,
    default_transform_v3,
    default_transform_v2,
    default_transform,
    transforms_eval,
)

BATCH_SIZE = 16

FN_MODEL1 = "./advinceptv3_fullconv_state.pth"
FN_MODEL2 = "./ens3incepv3_fullconv_state.pth"
FN_MODEL3 = "./ens4incepv3_fullconv_state.pth"
FN_MODEL4 = "./v34.epoch002"
FN_MODEL5 = "./v10_adv_densenet121_sgd.epoch010"
FN_MODEL6 = "./wide-resnet-50-2-export.hkl"
FN_MODEL7 = "./inceptionv4-97ef9c30.pth"
FN_MODEL8 = "./drn_c_58-0a53a92c.pth"
FN_MODEL9 = "./drn_d_105-12b40979.pth"
FN_MODEL10 = "./dpn107-extra.pth"
FN_MODEL11 = "./v62_adv_dpn131_sgd.epoch005"

DEFENSE_LIST = [
    'v27',
    'v28',
    'v29',
    'v36',
    'v10',
    'v8',
    'v19',
    'v51',
    'v52',
    'v55',
    'v67',
]


def _load_wideresnet_50_2_params():
    params = {}
    with tbl.open_file(FN_MODEL6, 'r') as store:
        for elem in store.get_node('/data_0'):
            name = elem._v_pathname[8:]
            ndarray_value = np.array(elem._v_children['data_0'])
            params[name] = Variable(
                torch.from_numpy(ndarray_value).cuda(),
                requires_grad=True)
    return params


class WideResNet50_2(object):
    def __init__(self, params):
        self.params = params

    def __call__(self, input_):
        params = self.params

        def conv2d(input, params, base, stride=1, pad=0):
            return F.conv2d(input, params[base + '.weight'],
                            params[base + '.bias'], stride, pad)

        def group(input, params, base, stride, n):
            o = input
            for i in range(0, n):
                b_base = ('%s.block%d.conv') % (base, i)
                x = o
                o = conv2d(x, params, b_base + '0')
                o = F.relu(o)
                o = conv2d(o, params, b_base + '1',
                           stride=(i == 0) and stride or 1,
                           pad=1)
                o = F.relu(o)
                o = conv2d(o, params, b_base + '2')
                if i == 0:
                    o += conv2d(x, params, b_base + '_dim',
                                stride=stride)
                else:
                    o += x
                o = F.relu(o)
            return o

        # determine network size by parameters
        blocks = [sum([re.match(
            'group%d.block\d+.conv0.weight' % j, k) is not None
            for k in params.keys()]) for j in range(4)]

        def f(input, params):
            o = F.conv2d(input,
                         params['conv0.weight'],
                         params['conv0.bias'],
                         2,
                         3)
            o = F.relu(o)
            o = F.max_pool2d(o, 3, 2, 1)
            o_g0 = group(o, params, 'group0', 1, blocks[0])
            o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
            o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
            o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
            o = F.avg_pool2d(o_g3, 7, 1, 0)
            o = o.view(o.size(0), -1)
            o = F.linear(o, params['fc.weight'], params['fc.bias'])
            return o

        return f(input_, self.params)


def _predict_top5(model_i, probs, num_inst, num_classes, st_idx):
    probs = F.softmax(probs)
    probs = probs.data.cpu().numpy().reshape(
        (num_inst, num_classes))

    topk_preds = []
    topk_probs = []
    for k in range(5):
        top1_preds = probs.argmax(axis=1).ravel()
        top1_probs = []
        for inst_idx in range(len(top1_preds)):
            p = probs[inst_idx][top1_preds[inst_idx]]
            p = float(p.ravel()[0])
            probs[inst_idx][top1_preds[inst_idx]] = 0
            top1_probs.append(p)
        top1_probs = np.array(top1_probs)
        if num_classes == 1000:
            top1_preds += 1
        topk_preds.append(top1_preds)
        topk_probs.append(top1_probs)

    defense_name = DEFENSE_LIST[model_i]
    rows = []
    for inst_idx in range(probs.shape[0]):
        row = dict(
            idx=st_idx + inst_idx,
            defense=defense_name,
            top1_pred=topk_preds[0][inst_idx],
            top2_pred=topk_preds[1][inst_idx],
            top3_pred=topk_preds[2][inst_idx],
            top4_pred=topk_preds[3][inst_idx],
            top5_pred=topk_preds[4][inst_idx],
            top1_prob=topk_probs[0][inst_idx],
            top2_prob=topk_probs[1][inst_idx],
            top3_prob=topk_probs[2][inst_idx],
            top4_prob=topk_probs[3][inst_idx],
            top5_prob=topk_probs[4][inst_idx],
        )
        rows.append(row)
    return rows


def predict(X, X2, X3, X4, models=[], start_idx=0):
    """
    Return the part of prediction (top1-5)

    Columns:
        * defense name
        * top1-5 predicted labels and probabilities
    """
    assert len(models) > 0

    ret_rows = []
    for model_i, m in enumerate(models):
        if model_i in [0, 1, 2, 3, 6]:
            probs = m(Variable(X, volatile=True))
            ret_rows += _predict_top5(
                model_i, probs, X.size()[0], 1001, start_idx)

        elif model_i in [4]:
            probs = m(Variable(X, volatile=True))
            ret_rows += _predict_top5(
                model_i, probs, X.size()[0], 1000, start_idx)

        elif model_i in [5]:
            probs = m(Variable(X2, volatile=True))
            ret_rows += _predict_top5(
                model_i, probs, X.size()[0], 1000, start_idx)

        elif model_i in [7, 8]:
            probs = m(Variable(X3, volatile=True))
            ret_rows += _predict_top5(
                model_i, probs, X.size()[0], 1000, start_idx)

        elif model_i in [9, 10]:
            preds = m(Variable(X4, volatile=True))
            ret_rows += _predict_top5(
                model_i, probs, X.size()[0], 1000, start_idx)

    return ret_rows


def _xgb_predict(df_base):
    df_feat = xgb_feat.gen_feat(df_base)

    # feature generation
    X = df_feat[xgb_feat.FEATURE_COLS].as_matrix()
    dmat = xgb.DMatrix(X)
    params = dict(
        objective='binary:logistic',
        eta=0.1,
        max_depth=6,
        silent=1,
        nthread=4,
        subsample=0.9,
        colsample_bylevel=0.5,
        reg_lambda=2,
        reg_alpha=1,
    )

    # xgb predict & get weights for majority voting
    for leave_attack_name in xgb_feat.ATTACK_LIST:
        bst_fn = './loocv_{}.bin'.format(
            leave_attack_name)
        bst = xgb.Booster(params)
        bst.load_model(bst_fn)
        df_feat.loc[:, 'pred_{}'.format(leave_attack_name)] = (
            bst.predict(dmat))

    # weighted majority voting
    df_feat.loc[:, 'weight'] = df_feat[[
        'pred_{}'.format(leave_attack_name)
        for attack_name in xgb_feat.ATTACK_LIST
    ]].mean(axis=1)

    df_feat = df_feat[[
        'defense',
        'name',
        'top1_pred',
        'weight',
    ]].sort_values(by=[
        'name',
        'defense',
    ])
    return df_feat


def main():
    input_dir = sys.argv[1]
    output_name = sys.argv[2]

    # NN1
    m1 = inception_v3_fullconv(transform_input=False, num_classes=1001)
    m1 = m1.cuda()
    m1.load_state_dict(torch.load(FN_MODEL1))
    m1.eval()
    # NN2
    m2 = inception_v3_fullconv(transform_input=False, num_classes=1001)
    m2 = m2.cuda()
    m2.load_state_dict(torch.load(FN_MODEL2))
    m2.eval()
    # NN3
    m3 = inception_v3_fullconv(transform_input=False, num_classes=1001)
    m3 = m3.cuda()
    m3.load_state_dict(torch.load(FN_MODEL3))
    m3.eval()
    # NN4
    m4 = inceptionresnetv2(
        num_classes=1001,
        pretrained='imagenet+background')
    m4 = torch.nn.DataParallel(m4).cuda()
    m4.load_state_dict(torch.load(FN_MODEL4)['state_dict'])
    m4 = m4.module
    m4.eval()
    # NN5
    m5 = densenet121(pretrained=False)
    m5 = torch.nn.DataParallel(m5).cuda()
    chk = torch.load(FN_MODEL5)
    m5.load_state_dict(chk['state_dict'])
    m5 = m5.module
    m5.eval()
    # NN6
    m6 = WideResNet50_2(_load_wideresnet_50_2_params())
    # NN7
    m7 = inceptionv4(
         num_classes=1001,
         pretrained='imagenet+background')
    m7 = m7.cuda()
    m7.load_state_dict(torch.load(FN_MODEL7))
    m7.eval()
    # NN8
    m8 = drn_c_58(pretrained=False)
    m8 = m8.cuda()
    m8.load_state_dict(torch.load(FN_MODEL8))
    m8.eval()
    # NN9
    m9 = drn_d_105(pretrained=False)
    m9 = m9.cuda()
    m9.load_state_dict(torch.load(FN_MODEL9))
    m9.eval()
    # NN10
    m10 = dpn107(
        num_classes=1000,
        pretrained=False,
        test_time_pool=True)
    m10 = m10.cuda()
    m10.load_state_dict(torch.load(FN_MODEL10))
    m10.eval()
    # NN11
    m11 = dpn131(
        num_classes=1000,
        pretrained=False,
        test_time_pool=True)
    m11 = torch.nn.DataParallel(m11).cuda()
    m11.load_state_dict(torch.load(FN_MODEL11)['state_dict'])
    m11 = m11.module
    m11.eval()

    # Dataset
    ds = Dataset(
        input_dir,
        transform=default_inception_transform(299))
    ds2 = Dataset(
        input_dir,
        transform=default_transform_v2(224))
    ds3 = Dataset(
        input_dir,
        transform=default_transform_v3(224))
    ds4 = Dataset(
        input_dir,
        transform=transforms_eval(320))

    # DataLoader
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    loader2 = DataLoader(ds2, batch_size=BATCH_SIZE, shuffle=False)
    loader3 = DataLoader(ds3, batch_size=BATCH_SIZE, shuffle=False)
    loader4 = DataLoader(ds4, batch_size=BATCH_SIZE, shuffle=False)

    loader2_iter = iter(loader2)
    loader3_iter = iter(loader3)
    loader4_iter = iter(loader4)

    preds = []
    filenames = []
    for idx, (X, y) in enumerate(loader):
        X = X.cuda()
        y = y.cuda()

        X2, y2 = loader2_iter.next()
        X2 = X2.cuda()

        X3, y3 = loader3_iter.next()
        X3 = X3.cuda()

        X4, y4 = loader4_iter.next()
        X4 = X4.cuda()

        preds += predict(
            X, X2, X3, X4,
            models=[m1, m2, m3, m4, m5, m6, m7, m8, m9,
                    m10, m11],
            start_idx=idx * BATCH_SIZE)
        filenames += loader.get_filenames(idx, len(X))

    df_feat_base = pd.DataFrame(preds)
    df_name = pd.DataFrame({
        'name': filenames,
        'idx': list(range(len(filenames))),
    })
    df_feat_base = df_feat_base.merge(
        df_name, how='left', on='idx')
    df_feat_base.drop('idx', axis=1, inplace=True)
    df_weight = _xgb_predict(df_feat_base)
    selected_models = [
        'v27',
        'v28',
        'v29',
        'v36',
        'v51',
        'v52',
        'v55',
        'v67',
    ]
    df_weight = df_weight[
        df_weight.defense.isin(selected_models)]

    preds = []
    for name, df_part in df_weight.groupby('name'):
        ct = Counter()
        for idx, row in df_part.iterrows():
            ct[row.top1_pred] += row.weight
        preds.append({
            'pred': ct.most_common(1)[0][0],
            'name': name,
        })

    pd.DataFrame(preds).to_csv(
        output_name,
        index=False,
        header=None)


if __name__ == '__main__':
    main()
