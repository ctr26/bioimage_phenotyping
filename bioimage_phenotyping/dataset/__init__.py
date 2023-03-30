from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import scale, power_transform, robust_scale, normalize
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV, RFE

import functools
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from .cellprofiler import Cellprofiler
import pandas as pd

import utils
# import bioimage_phenotyping as bip
from . import models

@pd.api.extensions.register_dataframe_accessor("bip")
class BioImageDataFrame:
    def __init__(self, df):
        self.df = df
    
    def drop_sigma(n=5):
        return cleaning.drop_sigma(self.df,n)
      
    def clean(self):
        return self.df.bip.drop_sigma(5).bip.model_rejection()
    
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
        return models.get_scoring_df(self.df,**locals())

    # @functools.
    # Needs python 3.9
    def get_shap_df(
        self,
        model,
        variable="Cell",
        groupby=None,
        augment=None,
        shap_samples=100,
        samples=None,
        *args,
        **kwargs,
    ):
        pass



    def get_score_report(
        self,
        variable="Cell",
        groupby=None,
        augment=None,
        model=RandomForestClassifier(),
    ):
        pass

    def get_score_df_from_model(self, model, variable, X_test, y_test):
        pass

    def drop_from_index(self, item):
        pass

    # df.index.names.difference(["Cell"])
    # @functools.cache
    def grouped_median(self, group="ObjectNumber"):
        pass

    def bootstrap(self, groups, size, group="ObjectNumber"):
        pass

    def drop_sigma(self, sigma=5, axis=0):
        pass

    def model_rejection(self,model=IsolationForest()):
        pass

    def clean(self):
        pass

    def preprocess(self, type="power_transform"):
        pass

    def groupby_conj(self, group):
        pass

    def groupby_counts(self, group):
        pass

    def summary_counts(self, name="ObjectNumber"):
        pass
    
    def simple_counts(self):
        pass

    # Augment not implemented
    def groupby_train_split(
        self, df, variable, groupby, frac=0.8, seed=42, augment=None
    ):
        pass

    def train_test_split(
        self, variable="Cell", frac=0.8, augment=None, groupby=None, seed=42
    ):
        pass
    
    def balance_dataset(self, variable):
        pass

    def select_features(self, variable="Drug"):
        pass

    def feature_importances(
        pass

    def keeplevel(self, level):
        pass

    # @functools.
    # Needs python 3.9
    # This is so confusing:
    def get_shap_df(
        self,
        model,
        variable="Cell",
        groupby=None,
        augment=None,
        shap_samples=100,
        samples=None,
        *args,
        **kwargs,
    ):
        return get_shap_df(
            self.get_shap_values(
                model,
                variable="Cell",
                groupby=None,
                augment=None,
                shap_samples=100,
                samples=None,
                *args,
                **kwargs,
            )
        )