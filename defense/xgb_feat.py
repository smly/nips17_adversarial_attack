# -*- coding: utf-8 -*-
from collections import Counter

import pandas as pd
import numpy as np


FEATURE_COLS = [
    # global_feat_mvbase4_voting
    'mv4_most_common',
    'mv4_second_most_common',

    # global_feat_mv11_voting
    'mv11_most_common',
    'mv11_second_most_common',

    # own_feat_prob
    'top1_prob',
    'top2_prob',
    'top3_prob',
    'top4_prob',
    'top5_prob',
    'first_second_probdiff',
    'second_to_five_sum',

    'defense_le',
    'defense_is_dpn',
    'defense_is_advincep',

    'is_in_dpn_top5_freq',
    'is_in_advincep_top5_freq',
]
ATTACK_LIST = [
    # 'v0',
    'v1',
    'v2',
    'v3',
    'v4',
    'v5',
    'v6',
    'v7',
    'v8',
    'v9',
    'v17',
    'v18',
    'v20',
    'v21',
    'v23',
    'v24',
    'v25',
    'v26',
    'v27',
    'v34',
    'v35',
    'v37',
    'v38',
    'v39',
    'v40',
    # 'v41',
    'v42',

    # 'v61',
    # 'v65',
]


def global_feat_mvbase4_voting(df):
    df_feat = []
    df = df[df.defense.isin(['v27', 'v28', 'v29', 'v36'])]

    df_gr = df[[
        'defense', 'name', 'top1_pred']].groupby('name')
    for image_name, row in df_gr:
        cnt = Counter(row.top1_pred)
        most_common_count = cnt.most_common(1)[0][1]
        second_most_count = 0
        if len(cnt) > 1:
            second_most_count = cnt.most_common(2)[1][1]

        df_feat.append(dict(
            name=image_name,
            mv4_most_common=most_common_count,
            mv4_second_most_common=second_most_count,
        ))
    df_feat = pd.DataFrame(df_feat)
    return df_feat[[
        'name',
        'mv4_most_common',
        'mv4_second_most_common',
    ]]


def global_feat_mv11_voting(df):
    df_feat = []
    df_gr = df[['defense', 'name', 'top1_pred']].groupby(
        'name')
    for image_name, row in df_gr:
        cnt = Counter(row.top1_pred)
        most_common_count = cnt.most_common(1)[0][1]
        second_most_count = 0
        if len(cnt) > 1:
            second_most_count = cnt.most_common(2)[1][1]

        df_feat.append(dict(
            name=image_name,
            mv11_most_common=most_common_count,
            mv11_second_most_common=second_most_count,
        ))
    df_feat = pd.DataFrame(df_feat)
    return df_feat[[
        'name',
        'mv11_most_common',
        'mv11_second_most_common',
    ]]


def own_feat_prob(df):
    df.loc[:, 'first_second_probdiff'] = df.top1_prob - df.top2_prob
    df.loc[:, 'top5_sum'] = df[[
        'top1_prob',
        'top2_prob',
        'top3_prob',
        'top4_prob',
        'top5_prob',
    ]].sum(axis=1)
    df.loc[:, 'second_to_five_sum'] = df[[
        'top2_prob',
        'top3_prob',
        'top4_prob',
        'top5_prob',
    ]].sum(axis=1)

    return df[[
        'defense',
        'name',

        'top1_prob',
        'top2_prob',
        'top3_prob',
        'top4_prob',
        'top5_prob',
        'first_second_probdiff',
        'second_to_five_sum',
    ]]


def own_feat_is_dpn_model(df):
    factorize_dict = {
        'v27': 0,
        'v28': 1,
        'v29': 2,
        'v51': 3,
        'v52': 4,
        'v55': 5,
        'v67': 6,
        'v36': 7,
        'v10': 8,
        'v19': 9,
        'v8': 10,
    }

    df.loc[:, 'defense_le'] = df.defense.apply(
        lambda x: factorize_dict[x])
    df.loc[:, 'defense_is_dpn'] = df.defense.isin(
        ['v55', 'v67']) * 1
    return df[[
        'defense',
        'name',

        'defense_is_dpn',
        'defense_le',
    ]]


def own_feat_is_adv_incep_baseline(df):
    df.loc[:, 'defense_is_advincep'] = df.defense.isin(
        ['v27', 'v28', 'v29', 'v36']) * 1
    return df[[
        'defense',
        'name',

        'defense_is_advincep',
    ]]


def global_feat_advincep_top1_in_dpn_top5(df_):
    advincep_model_names = ['v27', 'v28', 'v29', 'v36']
    dpn_model_names = ['v55', 'v67']
    df = df_.copy()

    # Find advincep top1 label
    df = df[df.defense.isin(advincep_model_names)]
    df_feat = []
    df_gr = df[[
        'defense',
        'name',
        'top1_pred',
    ]].groupby(['name'])
    for idx, row in df_gr:
        image_name = idx
        cnt = Counter(row.top1_pred)
        most_common_pred = cnt.most_common(1)[0][0]
        df_feat.append(dict(
            name=image_name,
            incep_top1_pred=most_common_pred,
        ))
    df_feat = pd.DataFrame(df_feat)

    # Is in dpn top5?
    df = df_.copy()
    df = df[df.defense.isin(dpn_model_names)]
    df = df[[
        'defense',
        'name',
        'top1_pred',
        'top2_pred',
        'top3_pred',
        'top4_pred',
        'top5_pred',
    ]].merge(df_feat, how='left', on='name')
    df_gr = df.groupby('name')
    df_feat2 = []
    for idx, row in df_gr:
        image_name = idx
        incep_lbl = row.iloc[0]['incep_top1_pred']

        cnt = Counter(row.top1_pred)
        cnt.update(Counter(row.top2_pred))
        cnt.update(Counter(row.top3_pred))
        cnt.update(Counter(row.top4_pred))
        cnt.update(Counter(row.top5_pred))

        freq = cnt.get(incep_lbl, 0)
        df_feat2.append(dict(
            name=image_name,
            is_in_dpn_top5_freq=freq,
        ))
    df_feat2 = pd.DataFrame(df_feat2)
    return df_feat2[[
        'name',
        'is_in_dpn_top5_freq',
    ]]


def global_feat_dpn_top1_in_advincep_top5(df_):
    advincep_model_names = ['v27', 'v28', 'v29', 'v36']
    dpn_model_names = ['v55', 'v67']
    df = df_.copy()

    # Find dpn top1 label
    df = df[df.defense.isin(dpn_model_names)]
    df_feat = []
    df_gr = df[[
        'defense',
        'name',
        'top1_pred',
    ]].groupby('name')
    for idx, row in df_gr:
        image_name = idx
        cnt = Counter(row.top1_pred)
        most_common_pred = cnt.most_common(1)[0][0]
        df_feat.append(dict(
            name=image_name,
            dpn_top1_pred=most_common_pred,
        ))
    df_feat = pd.DataFrame(df_feat)

    # Is in advincep top5?
    df = df_.copy()
    df = df[df.defense.isin(advincep_model_names)]
    df = df[[
        'defense',
        'name',
        'top1_pred',
        'top2_pred',
        'top3_pred',
        'top4_pred',
        'top5_pred',
    ]].merge(df_feat, how='left', on='name')
    df_gr = df.groupby('name')
    df_feat2 = []
    for idx, row in df_gr:
        image_name = idx
        dpn_lbl = row.iloc[0]['dpn_top1_pred']

        cnt = Counter(row.top1_pred)
        cnt.update(Counter(row.top2_pred))
        cnt.update(Counter(row.top3_pred))
        cnt.update(Counter(row.top4_pred))
        cnt.update(Counter(row.top5_pred))

        freq = cnt.get(dpn_lbl, 0)
        df_feat2.append(dict(
            name=image_name,
            is_in_advincep_top5_freq=freq,
        ))
    df_feat2 = pd.DataFrame(df_feat2)
    return df_feat2[[
        'name',
        'is_in_advincep_top5_freq',
    ]]


def gen_feat(df_base):
    df_fe1 = global_feat_mvbase4_voting(df_base)
    df_fe2 = global_feat_mv11_voting(df_base)
    df_fe3 = own_feat_prob(df_base)
    df_fe4 = own_feat_is_dpn_model(df_base)
    df_fe5 = own_feat_is_adv_incep_baseline(df_base)
    df_fe6 = global_feat_advincep_top1_in_dpn_top5(df_base)
    df_fe7 = global_feat_dpn_top1_in_advincep_top5(df_base)

    df = df_base[[
        'defense',
        'name',
        'top1_pred',
        'top2_pred',
        'top3_pred',
        'top4_pred',
        'top5_pred',
    ]]
    df = df.merge(df_fe1, how='left', on='name')
    df = df.merge(df_fe2, how='left', on='name')
    df = df.merge(df_fe3, how='left', on=[
        'defense', 'name'])
    df = df.merge(df_fe4, how='left', on=[
        'defense', 'name'])
    df = df.merge(df_fe5, how='left', on=[
        'defense', 'name'])
    df = df.merge(df_fe6, how='left', on='name')
    df = df.merge(df_fe7, how='left', on='name')

    return df
