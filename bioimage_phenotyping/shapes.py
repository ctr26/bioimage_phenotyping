import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import SelectFromModel, RFECV, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import model_selection
import pathlib
import scipy
import random
import warnings
from tqdm import tqdm
import dask
from dask import delayed
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from distributed import Client
from bioimage_phenotyping import Cellprofiler
from sklearn.metrics import pairwise
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

def augment_at_theta(df, function, i, theta):
    return (
        df.apply(function, axis=1, theta=theta)
        .assign(augmentation=i, angle=theta)
        .set_index(["augmentation", "angle"], append=True)
    )


def dask_augment_df_alt(df, function, fold=0):
    #
    fold_sequence = np.append(np.array(0), np.random.uniform(0, 2 * np.pi, fold))
    return [
        delayed(augment_at_theta).apply(df=df, function=function, i=i, theta=theta)
        for i, theta in enumerate(fold_sequence)
    ]


def dask_augment_df(df, function, fold=0):
    #
    fold_sequence = np.append(np.array(0), np.random.uniform(0, 2 * np.pi, fold))
    return [
        delayed(df)
        .apply(function, axis=1, theta=theta)
        .assign(augmentation=i, angle=theta)
        .set_index(["augmentation", "angle"], append=True)
        for i, theta in enumerate(fold_sequence)
    ]


def rotate_control_points(series, theta=0):
    series_x, series_y = series.pipe(flat_series_to_dx_dy)
    series_x_prime = series_x * np.cos(theta) - np.array(series_y * np.sin(theta))
    series_y_prime = np.array(series_x * np.sin(theta)) + series_y * np.cos(theta)
    # series_prime = pd.concat([series_x_prime,series_y_prime],axis=1).assign(Angle=theta)
    series_prime = pd.concat([series_x_prime, series_y_prime])
    return series_prime.reindex(series.index, axis=1)


# import numba
# @numba.jit
def rotate_control_points_np(array, theta=0):
    odds = np.arange(0, len(array) - 1, 2)
    evens = np.arange(1, len(array), 2)
    x, y = array[odds], array[evens]
    x_prime = x * np.cos(theta) - np.array(y * np.sin(theta))
    y_prime = np.array(x * np.sin(theta)) + y * np.cos(theta)
    array_prime = array.copy()
    array_prime[odds] = x_prime
    array_prime[evens] = y_prime
    return array_prime


def align_coords_to_origin(series):
    # Series only now
    series_x, series_y = series.pipe(flat_series_to_dx_dy)
    series_x_prime = series_x - np.mean(series_x)
    series_y_prime = series_y - np.mean(series_y)
    series_prime = pd.concat([series_x_prime, series_y_prime], axis=0)
    return series_prime.reindex(series.index, axis=1)


def align_coords_to_origin_np(array):
    # Series only now
    odds = np.arange(0, len(array) - 1, 2)
    evens = np.arange(1, len(array), 2)

    x, y = array[odds], array[evens]

    x_prime = x - np.mean(x)
    y_prime = y - np.mean(y)

    array_prime = array.copy()

    array_prime[odds] = x_prime
    array_prime[evens] = y_prime

    return array_prime


def flat_series_to_dx_dy(series):
    # Series only now
    odds = np.arange(0, len(series.index) - 1, 2)
    evens = np.arange(1, len(series.index), 2)

    series_x = series[odds.astype(str)]
    series_y = series[evens.astype(str)]

    return series_x, series_y


def df_add_augmentation_index(df, index_name="augmentation"):
    return df.set_index(
        df.groupby(level=df.index.names).cumcount().rename(index_name), append=True
    )


def augment_distance_matrix(df, axis=0):
    return pd.concat(
        [
            # Numpy roll each row by i for each column
            df.transform(np.roll, 1, i, 0)
            for i in range(len(df.columns))
        ],
        axis=axis,
    )


def augment_repeat(df, fold=1):
    return df.reindex(df.index.repeat(fold))

def series_to_distance_matrix(x):
    distance_matrix = np.tril(
        pairwise.euclidean_distances(np.array([x[0::2], x[1::2]]).T)
    )
    return distance_matrix


def df_to_distance_matrix(df):
    return (
        df.apply(
            lambda x: np.tril(
                pairwise.euclidean_distances(np.array([x[0::2], x[1::2]]).T)
            )
            .flatten()
            .flatten(),
            axis=1,
            result_type="expand",
        )
        .replace(0, np.nan)
        .dropna(axis=1)
    )

#%TODO Number of bins matters
def df_to_distogram(df,hist_range=(0,1)):
    return df_to_distance_matrix(df).apply(
        lambda x: np.histogram(x, bins=len(x),range=range))[0], axis=1, result_type="expand")

def distance_matrix_to_cyclic_distograms(distmat,hist_range=(0,1)):
    cyclic_distograms = []
    
    distmat = np.array(distmat)
    distmat_shape = np.sqrt(len(distmat)).astype('int')
    distmat = np.reshape(distmat, (distmat_shape, distmat_shape))
    
    for i in range(1,len(distmat)//2 + 1):
        a = [np.diagonal(distmat, -i).copy(), np.diagonal(distmat, len(distmat)-i).copy()]
        a = [item for sublist in a for item in sublist]
        a = np.array(a)
        
        distogram = np.histogram(a, bins = len(a),range=hist_range)[0]
        cyclic_distograms.append(distogram)
        
    cyclic_distograms = [item for sublist in cyclic_distograms for item in sublist]
    cyclic_distograms = np.array(cyclic_distograms)
    return cyclic_distograms
    
def df_to_cyclic_distograms(df):
    df =  df.apply(
            lambda x: pairwise.euclidean_distances(np.array([x[0::2], x[1::2]]).T)            
            .flatten(),
            axis=1,
            result_type="expand",
        )
    
    df =  df.apply(
            lambda x: distance_matrix_to_cyclic_distograms(x)            
            .flatten(),
            axis=1,
            result_type="expand",
        )
    return df


# for augmentation in np.linspace(1,200,10).astype(int):
def augment_df_dask(df, function, fold=0):
    #
    fold_sequence = np.append(np.array(0), np.random.uniform(0, 2 * np.pi, fold))
    index = df.index
    df_no_index = dd.from_pandas(df.reset_index(drop=True), npartitions=len(df))
    # print(df_no_index)

    return pd.concat(
        [
            df_no_index.apply(function, axis=1, theta=theta)
            .assign(augmentation=i, angle=theta)
            .compute()
            .set_index(index, append=True)
            for i, theta in enumerate(fold_sequence)
        ]
    ).set_index(["augmentation", "angle"], append=True)


# for augmentation in np.linspace(1,200,10).astype(int):
def angular_augment_df(df, function, fold=0):
    #
    fold_sequence = np.append(np.array(0), np.random.uniform(0, 2 * np.pi, fold))
    # print(df_no_index)

    return pd.concat(
        [
            df.apply(function, theta=0, axis=1, raw=True)
            .assign(augmentation=i, angle=0)
            .set_index(["augmentation", "angle"], append=True)
            for i, theta in enumerate(fold_sequence)
        ]
    )


def angular_augment_X_y(X, y, function=rotate_control_points_np, fold=0):
    X_aug = angular_augment_df(X, function, fold)
    y_aug = angular_augment_df(y, lambda x, theta: x, fold)
    return X_aug, y_aug


def angular_augment_X_y_fun(function=rotate_control_points_np, fold=0):
    def return_fun(X, y):
        return angular_augment_X_y(X, y, function=rotate_control_points_np, fold=fold)

    return return_fun


def get_score_report_per_level(df, level="Features"):
    return (
        df.groupby(level="Features")
        .apply(
            lambda df: df.cellesce.grouped_median()
            .dropna(axis=1)
            .cellesce.get_score_report(variable="Cell")
        )
        .reset_index()
        .set_index("Metric")
        .loc[["f1-score", "recall", "precision"]]
        .reset_index()
    )


# x = df_splinedist.iloc[:, np.arange(0, len(df_splinedist.columns) - 1, 2)]
# y = df_splinedist.iloc[:, np.arange(1, len(df_splinedist.columns), 2)]
# plt.figure()
# plt.scatter(x.iloc[0], y.iloc[0])
# if not TEST_ROT:
#     plt.close()
# # %%
# df_splinedist_rot = df_splinedist.apply(
#     shapes.rotate_control_points_np, theta=-np.pi / 2, axis=1, raw=True
# )

# x = df_splinedist_rot.iloc[:, np.arange(0, len(df_splinedist.columns) - 1, 2)]
# y = df_splinedist_rot.iloc[:, np.arange(1, len(df_splinedist.columns), 2)]
# plt.figure()
# plt.scatter(x.iloc[0], y.iloc[0])
# if not TEST_ROT:
#     plt.close()



class DistanceMatrix(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X, y=None, **fit_params):
        return self
    
    def transform(self,X, y=None, **fit_params):
        X_trans = (
            pd.DataFrame(X.copy())
            .pipe(df_to_distance_matrix))

        return X_trans
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y, **fit_params)


class Distogram(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X, y=None, **fit_params):
        return self
    
    def transform(self,X, y=None, **fit_params):
        X_trans = (
            pd.DataFrame(X.copy())
            .pipe(df_to_distogram))

        return X_trans
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y, **fit_params)

