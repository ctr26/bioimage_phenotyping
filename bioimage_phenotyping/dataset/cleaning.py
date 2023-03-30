from sklearn.ensemble import IsolationForest
import numpy as np
from scipy import stats
# TODO make general


def model_inlier_rejection(df,model=IsolationForest()):
    return df.loc[model.fit(df.values).predict(df) == 1]


def drop_sigma(df, sigma=5, axis=0):
    return df.mask(
        df.apply(lambda df: (np.abs(stats.zscore(df)) > sigma))
    ).dropna(axis=axis)