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
from sklearn.ensemble import BaggingClassifier, IsolationForest, RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    PowerTransformer,
    StandardScaler,
    normalize,
    power_transform,
    robust_scale,
    scale,
)

from .. import features, models, transforms
from . import cleaning, utils
from .cellprofiler import Cellprofiler


@pd.api.extensions.register_dataframe_accessor("bip")
class BioImageDataFrame:
    seed = 42

    def __init__(self, df, seed=42):
        self.df = df
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed

    def drop_sigma(self, n=5):
        return cleaning.drop_sigma(self.df, n)

    def clean(self):
        return self.df.bip.drop_sigma(5).bip.model_inlier_rejection()

    def drop_from_index(self, item):
        return utils.drop_from_index(self.df, item)

    def drop_from_list(self, list_in, item):
        return utils.drop_from_list(list_in, item)

    def get_scoring_df(
        self,
        variable="Cell",
        model=RandomForestClassifier(),
        kfolds=5,
        groupby=None,
        augment=None,
    ):
        return models.get_scoring_df(
            self.df,
            variable=variable,
            model=model,
            kfolds=kfolds,
            groupby=groupby,
            augment=augment,
        )

    # @functools.
    # Needs python 3.9
    # def get_shap_df(
    #     self,
    #     model,
    #     variable="Cell",
    #     groupby=None,
    #     augment=None,
    #     shap_samples=100,
    #     samples=None,
    #     *args,
    #     **kwargs,
    # ):
    #     pass

    def get_score_report(
        self,
        variable="Cell",
        groupby=None,
        augment=None,
        model=RandomForestClassifier(),
    ):
        return models.get_score_report(
            self.df,
            variable=variable,
            groupby=groupby,
            augment=augment,
            model=model,
        )

    def get_score_df_from_model(self, model, variable, X_test, y_test):
        return models.score_df_from_model(model, variable, X_test, y_test)
        # return models.score_df_from_model(
        #     self.df, model=model, variable=variable, X_test=X_test, y_test=y_test
        # )

    # df.index.names.difference(["Cell"])
    # @functools.cache
    def grouped_median(self, group="ObjectNumber"):
        return utils.grouped_median(self.df, group)

    def bootstrap(self, groups, size, group="ObjectNumber"):
        pass

    def model_inlier_rejection(self, model=IsolationForest()):
        return cleaning.model_inlier_rejection(self.df, model=model)

    def preprocess(self, mode="power_transform"):
        return transforms.preprocess(self.df, mode=mode)

    def groupby_conj(self, group):
        return utils.groupby_conj(self.df, group)

    def groupby_counts(self, group):
        return utils.groupby_counts(self.df, group)

    def summary_counts(self, name="ObjectNumber"):
        pass

    def simple_counts(self):
        return utils.simple_counts(self.df)

    # Augment not implemented
    def groupby_train_split(self, variable, groupby, frac=0.8, seed=42):
        return models.groupby_train_split(
            self.df,
            variable=variable,
            groupby=groupby,
            frac=frac,
            seed=seed,
        )

    def train_test_split(
        self,
        variable="Cell",
        frac=0.8,
        augment=None,
        groupby=None,
    ):
        return models.train_test_split(
            self.df,
            variable=variable,
            frac=frac,
            augment=augment,
            groupby=groupby,
            seed=self.seed,
        )

    def balance_dataset(self, variable):
        return utils.balance_dataset(self.df, variable)

    def select_features(self, variable="Drug", model=RandomForestClassifier()):
        return features.select_from_model(self.df, variable=variable, model=model)

    def feature_importance(
        self,
        model_class=RandomForestClassifier,
        variable="Cell",
        groupby=None,
        augment=None,
        kfolds=1,
    ):
        return features.importance.leaf_model(
            self.df,
            model_class=model_class,
            variable=variable,
            groupby=groupby,
            augment=augment,
            kfolds=kfolds,
        )

    def keeplevel(self, level):
        return utils.keeplevel(self.df, level)

    # # @functools.
    # # Needs python 3.9
    # # This is so confusing:
    # def get_shap_df(
    #     self,
    #     model,
    #     variable="Cell",
    #     groupby=None,
    #     augment=None,
    #     shap_samples=100,
    #     samples=None,
    #     *args,
    #     **kwargs,
    # ):
    #     return get_shap_df(
    #         self.get_shap_values(
    #             model,
    #             variable="Cell",
    #             groupby=None,
    #             augment=None,
    #             shap_samples=100,
    #             samples=None,
    #             *args,
    #             **kwargs,
    #         )
    #     )
