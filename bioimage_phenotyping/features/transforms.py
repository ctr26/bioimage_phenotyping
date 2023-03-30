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
        "standard": lambda x: scale(x),
        "robust_scale": lambda x: robust_scale(x),
        "normalize": lambda x: normalize(x),
    }

def preprocess(df, type="power_transform"):

    preprocessor = preprocessor_lookup[type]
    # scaled_df = pd.DataFrame(
    #     preprocessor(self), index=self.index, columns=self.columns
    # )
    df = pd.DataFrame(
        preprocessor(self.df),
        index=self.df.index,
        columns=self.df.columns,
    )
    return df