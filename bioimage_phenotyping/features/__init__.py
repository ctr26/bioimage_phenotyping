from . import pca,plotting,importance
from .. import models

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


def select_from_model(df, variable="Drug",model=RandomForestClassifier()):
    pipe = Pipeline(
        [("PCA", PCA()), ("modelselect", SelectFromModel(model))]
    )
    X_train, X_test, y_train, y_test = (
        # .balance_dataset(variable)
        models.train_test_split(df,variable)
    )
    pipe.fit(X_train.values, y_train)
    return pd.DataFrame(pipe.transform(df), index=df.index)
