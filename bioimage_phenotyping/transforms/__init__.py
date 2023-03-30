import functools

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, PowerTransformer,
                                   StandardScaler, normalize, power_transform,
                                   robust_scale, scale)

preprocessor_lookup = {
    "power_transform": lambda x: power_transform(x, method="yeo-johnson"),
    "standard": scale,
    "robust_scale": robust_scale,
    "normalize": normalize,
}


def preprocess(df, mode="power_transform"):
    # preprocessor = preprocessor_lookup[mode
    # scaled_df = pd.DataFrame(
    #     preprocessor(self), index=self.index, columns=self.columns
    # )
    df = pd.DataFrame(
        preprocessor_lookup[mode](df),
        index=df.index,
        columns=df.columns,
    )
    return df
