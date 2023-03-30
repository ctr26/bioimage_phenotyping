import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import (BaggingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, PowerTransformer,
                                   StandardScaler, normalize, power_transform,
                                   robust_scale, scale)

def keeplevel(df, level):
    return df.droplevel(df.pipe(drop_from_index,level))

def simple_counts(df):
    return df.count().iloc[0]

def grouped_median(df, group="ObjectNumber"):
    return df.groupby(level=drop_from_index(df, group)).median()


def drop_from_index(df, item):
    return drop_from_list(list(df.index.names), item)


def drop_from_list(list_in, item):
    item = [item] if isinstance(item, str) else item
    return list(set(list_in) - set(item))


def balance_dataset(df, variable):
    g = df.groupby(level=variable, group_keys=False)
    df = g.apply(lambda x: x.sample(g.size().min()))
    print(df.index.get_level_values(variable).value_counts())
    return df

def drop_sigma(df, sigma=5, axis=0):
    return df.mask(
        df.apply(lambda df: (np.abs(stats.zscore(df)) > sigma))
    ).dropna(axis=axis)
    

# TODO fix or implement
def bootstrap(df, groups, size, group="ObjectNumber"):
    # boostrap_df = df.groupby(level=list(set(groups) - {group})).median()
    # random_sample, make n groups, median of each group
    return NotImplementedError 


# 2DO fix
def groupby_not(df,groups,group):
    return df.groupby(level=list(set(groups) - {group}))


def groupby_conj(df, group):
    return df.groupby(
        level=df.pipe(drop_from_index,group), group_keys=False
    )

def groupby_counts(df, group):
    return df.pipe(groupby_conj,group).size()


def unique_levels(df, variable):
    return df.index.get_level_values(variable).to_series().unique()