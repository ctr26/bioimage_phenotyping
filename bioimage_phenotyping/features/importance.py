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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, PowerTransformer,
                                   StandardScaler, normalize, power_transform,
                                   robust_scale, scale)

from .. import models


# TODO, typee hinting
def get_shap_df(shaps):
    return (
        pd.DataFrame(shaps.values, columns=shaps.feature_names)
        .rename_axis("Sample")
        .reset_index()
        .melt(id_vars="Sample", var_name="Feature", value_name="Shap Value")
    )


def get_shap_values(
    df,
    model,
    variable="Cell",
    groupby=None,
    augment=None,
    shap_samples=100,
    samples=None,
    *args,
    **kwargs,
):
    # X = np.array(df.apply(pd.to_numeric))
    X, y = df, list(df.index.get_level_values(variable))
    X100 = shap.utils.sample(np.array(X), 100)

    # y = df.index.get_level_values(variable)
    # y = DistanceMatrix().fit_transform(X)

    X_train, X_test, y_train, y_test = df.apply(pd.to_numeric).bip.train_test_split(
        variable, groupby=groupby, augment=augment
    )
    # model = RandomForestClassifier()
    # model = Pipeline([('Distogram', Distogram()),
    #                 ('scaler', StandardScaler()),
    #                 ('rf', RandomForestClassifier())])

    y_train = LabelEncoder().fit(y).transform(y_train)
    y_test = LabelEncoder().fit(y).transform(y_test)

    model.fit(X_train.values, y_train)
    model.score(X_test.values, y_test)

    explainer = shap.Explainer(model.predict, X100, *args, **kwargs)

    shap_values = explainer(X)
    return shap_values



def leaf_model(
    df,
    model_class=RandomForestClassifier,
    variable="Cell",
    groupby=None,
    augment=None,
    kfolds=1,
):
    importance_list = []

    for fold in range(1, kfolds + 1):
        model = model_class()
        X_train, X_test, y_train, y_test = models.train_test_split(df, variable, groupby=groupby, augment=augment, seed=fold)

        model.fit(X_train.values, y_train)

        print(classification_report(y_test, model.predict(X_test)))
        print(metrics.cohen_kappa_score(y_test, model.predict(X_test)))

        importance = (
            pd.DataFrame(
                model.feature_importances_,
                index=pd.Series(X_train.columns, name="Feature"),
                columns=["Importance"],
            )
            .assign(Fold=fold)
            .sort_values(ascending=False, by="Importance")
        )

        importance["Cumulative Importance"] = importance.cumsum()["Importance"]
        importance_list.append(importance)
        # scores = self.get_score_df_from_model(model,variable,X_test,y_test)

    return pd.concat(importance_list)
