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
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV, RFE

import functools


import os

#  %%
# https://www.nature.com/articles/nmeth.4397
# Graphs to generate
# Pairwise euclidean
# New dose response using powerTransformer
# AUR ROC Curves for confusion matrices
# COnfusion matrix for classifying cells on test data

# [TODO: show scores for each training, AUR, F, ACCURACY as a grid]

#  I AM VERY AWARE OF HOW POOR THE CODE
#  IS, AND HOW URGENTLY IT NEEDS REFRACTORING

# %%
# from sklearn.metrics import homogeneity_score
from sklearn.cluster import KMeans
import numpy.matlib

# from scipy.stats import kstest
# import scipy.stats
import json

from sklearn.utils import check_matplotlib_support
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("bip")
class CellprofilerDataFrame:
    def __init__(self, df):
        self.df = df

    def drop_from_list(self, list_in, item):
        item = [item] if isinstance(item, str) else item
        return list(set(list_in) - set(item))

    def get_scoring_df(
        self,
        variable="Cell",
        model=RandomForestClassifier(),
        kfolds=5,
        groupby=None,
        augment=None,
    ):
        # score_list = []
        # for fold in range(1,kfolds+1):
        #     score = (self.df.bip.get_score_report(variable, model)
        #              .assign(Fold=fold))
        #     score_list.append(score)
        return pd.concat(
            [
                (
                    self.df.bip.get_score_report(
                        variable=variable, model=model, groupby=groupby, augment=augment
                    ).assign(Fold=fold)
                )
                for fold in range(1, kfolds + 1)
            ]
        )

    # @functools.
    # Needs python 3.9
    def get_score_report(
        self,
        variable="Cell",
        groupby=None,
        augment=None,
        model=RandomForestClassifier(),
    ):
        # labels, uniques = pd.factorize(df.reset_index()[variable])
        df = self.df
        X, y = df, list(df.index.get_level_values(variable))
        uniques = df.index.get_level_values(variable).to_series().unique()
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
        X_train, X_test, y_train, y_test = df.bip.train_test_split(
            variable, groupby=groupby, augment=augment
        )
        model.fit(X_train, y_train)
        return self.get_score_df_from_model(model, variable, X_test, y_test)
        # # y_pred = pd.Series(model.predict(X_test), index=X_test.index)
        # scoring = X_test.apply(lambda x: model.predict([x])[0], axis=1).reset_index(
        #     variable, name="y_pred"
        # )

        # # scoring.groupby("Drug").
        # # %% Fix this
        # ck = (
        #     scoring.groupby(variable, group_keys=False)
        #     .apply(lambda x: metrics.cohen_kappa_score(x["y_pred"], x[variable]))
        #     .rename("Cappa Kohen")
        # )
        # report = (
        #     pd.DataFrame(
        #         metrics.classification_report(
        #             y_test, model.predict(X_test), output_dict=True
        #         )
        #     )
        #     .rename_axis("Metric")
        #     .drop(["accuracy", "macro avg", "weighted avg"], axis=1)
        #     .append(ck)
        # )

        # report_tall = (
        #     report.melt(
        #         # id_vars="Metric",
        #         var_name="Kind",
        #         value_name="Score",
        #         ignore_index=False,
        #     )
        #     .assign(**{"Variable": variable})
        #     .reset_index()
        # )

        # return report_tall

    def get_score_df_from_model(self, model, variable, X_test, y_test):
        scoring = X_test.apply(lambda x: model.predict([x])[0], axis=1).reset_index(
            variable, name="y_pred"
        )

        # scoring.groupby("Drug").
        # %% Fix this
        ck = (
            scoring.groupby(variable, group_keys=False)
            .apply(lambda x: metrics.cohen_kappa_score(x["y_pred"], x[variable]))
            .rename("Cappa Kohen")
        )
        report = (
            pd.DataFrame(
                metrics.classification_report(
                    y_test, model.predict(X_test), output_dict=True
                )
            )
            .rename_axis("Metric")
            .drop(["accuracy", "macro avg", "weighted avg"], axis=1)
            .append(ck)
        )

        report_tall = (
            report.melt(
                # id_vars="Metric",
                var_name="Kind",
                value_name="Score",
                ignore_index=False,
            )
            .assign(**{"Variable": variable})
            .reset_index()
        )
        return report_tall

    def drop_from_index(self, item):
        return self.df.bip.drop_from_list(list(self.df.index.names), item)

    # df.index.names.difference(["Cell"])
    # @functools.cache
    def grouped_median(self, group="ObjectNumber"):
        return (self
                .df.groupby(level=self.df.bip.drop_from_index(group))
                .median())

    def bootstrap(self, groups, size, group="ObjectNumber"):
        self.groupby(level=list(set(self.attrs["index_headers"]) - {group})).median()
        # random_sample, make n groups, median of each group
        pass

    def drop_sigma(self, sigma=5, axis=0):
        return self.df.mask(
            self.df.apply(lambda df: (np.abs(stats.zscore(df)) > sigma))
        ).dropna(axis=axis)

    def isolation_forest(self):
        return self.df.loc[IsolationForest().fit(self.df).predict(self.df) == 1]

    def clean(self):
        return self.df.bip.drop_sigma(5).bip.isolation_forest()

    def preprocess(self, type="power_transform"):
        preprocessor_lookup = {
            "power_transform": lambda x: power_transform(x, method="yeo-johnson"),
            "standard": lambda x: scale(x),
            "robust_scale": lambda x: robust_scale(x),
            "normalize": lambda x: normalize(x),
        }
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

    def groupby_conj(self, group):
        return self.df.groupby(
            level=self.df.bip.drop_from_index(group), group_keys=False
        )

    def groupby_counts(self, group):
        return self.df.bip.groupby_conj(group).size()

    # def summary_counts(self, name="ObjectNumber"):
    #     return (
    #         self.groupby(
    #         level=self.drop_from_index(name))
    #             .size()
    #     )

        
    def simple_counts(self):
        return self.df.count().iloc[0]

    # Augment not implemented
    def groupby_train_split(
        self, df, variable, groupby, frac=0.8, seed=42, augment=None
    ):
        split_list = []
        for group_name, df_group in df.groupby(
            groupby, sort=False, as_index=False, group_keys=False
        ):

            X = (
                df_group.apply(lambda x: x)
                .sort_index()
                .sample(frac=frac, random_state=seed)
            )
            y = X.index.to_frame()[[variable]].astype(str)

            split_list.append(
                model_selection.train_test_split(X, y, stratify=y, random_state=seed)
            )
        X_train, X_test, y_train, y_test = tuple(map(pd.concat, zip(*split_list)))
        return X_train, X_test, y_train, y_test

    def train_test_split(
        self, variable="Cell", frac=0.8, augment=None, groupby=None, seed=42
    ):
        # df = self.df.sample(frac=1,random_state=seed)
        df = self.df

        # g = df.groupby(level=variable,group_keys=False)
        # df = g.apply(lambda x: x.sample(g.size().min()));df

        # labels = df.reset_index()[[variable]].astype(str)

        # This stops the model cheating
        # y = df.reset_index()[[variable]].astype(str)
        # return model_selection.train_test_split(df,y,stratify=y)

        # X_train = df.groupby(groupby, as_index=False,group_keys=False).apply(
        #     lambda x: x.sample(frac=frac)
        # )

        # groups = df.groupby(groupby, sort=False, as_index=False, group_keys=False)
        # group_0 = groups.get_group(list(groups.groups)[0])
        # group_1 = groups.get_group(list(groups.groups)[1])

        # gss = GroupShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
        # groups = df.index.get_level_values(groupby)
        X = df
        y = df.index.to_frame()[[variable]].astype(str)
        # (X_train_idx,X_test_idx), (y_train_idx,y_test_idx) = gss.split(X, y, groups)
        if groupby is not None:
            return self.groupby_train_split(
                df, variable, groupby, frac=0.8, seed=42, augment=augment
            )

        if augment is not None:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, stratify=y, random_state=seed
            )
            X_train, y_train = augment(X_train, y_train)
            return X_train, X_test, y_train, y_test

        return model_selection.train_test_split(X, y, stratify=y, random_state=seed)

        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        return X_train, X_test, y_train, y_test

        X_train = (
            df.groupby(groupby, sort=False, as_index=False, group_keys=False)
            .sample(frac=1, random_state=seed)
            .sort_index()
            .sample(frac=0.8, random_state=seed)
        )

        if len(df) == len(X_train):
            X_train = df.sample(frac=frac, random_state=seed).sort_index()

        dupe_df = pd.concat([df, X_train])
        X_test = dupe_df[~dupe_df.index.duplicated(keep=False)]
        y_train = X_train.index.to_frame()[[variable]].astype(str)
        y_test = X_test.index.to_frame()[[variable]].astype(str)
        # feature_df_in = feature_df_median_in
        return X_train, X_test, y_train, y_test

    def balance_dataset(self, variable):
        g = self.groupby(level=variable, group_keys=False)
        df = g.apply(lambda x: x.sample(g.size().min()))
        print(df.index.get_level_values(variable).value_counts())
        return df

    def select_features(self, variable="Drug"):
        pipe = Pipeline(
            [("PCA", PCA()), ("modelselect", SelectFromModel(RandomForestClassifier()))]
        )
        X_train, X_test, y_train, y_test = (
            self
            # .balance_dataset(variable)
            .train_test_split(variable)
        )
        pipe.fit(X_train, y_train)
        df = pd.DataFrame(pipe.transform(self.df), index=self.df.index)
        df.attrs.update(self.df.attrs)
        return df

    def feature_importances(
        self,
        model_class=RandomForestClassifier,
        variable="Cell",
        groupby=None,
        augment=None,
        kfolds=1,
    ):
        importance_list = []

        for fold in range(1, kfolds + 1):
            model = model_class()
            X_train, X_test, y_train, y_test = (
                self.df
                # .balance_dataset(variable)
                .bip.train_test_split(
                    variable, groupby=groupby, augment=augment, seed=fold
                )
            )
            model.fit(X_train, y_train)

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
            importance.attrs.update(self.df.attrs)
            importance_list.append(importance)
            # scores = self.get_score_df_from_model(model,variable,X_test,y_test)

        return pd.concat(importance_list)

    def keeplevel(self, level):
        return self.df.droplevel(self.df.bip.drop_from_index(level))

    # def get_score_report(df, model=RandomForestClassifier(), variable="Cell"):
    #     # labels, uniques = pd.factorize(df.reset_index()[variable])
    #     X, y = df, list(df.index.get_level_values(variable).astype("category"))
    #     # uniques = set(y)
    #     X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    #     X_train, X_test, y_train, y_test = df.bip.train_test_split(variable)
    #     model.fit(X_train, y_train)
    #     # y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    #     report = pd.DataFrame(
    #         metrics.classification_report(y_test, model.predict(X_test),
    #                                     output_dict=True)
    #     )
    #     report_tall = (
    #         report[set(y)]
    #         .rename_axis("Metric")
    #         .melt(
    #             # id_vars="Metric",
    #             var_name="Kind",
    #             value_name="Score",
    #             ignore_index=False,
    #         )
    #         .assign(**{"Variable": variable})
    #         .reset_index()
    #     )
    #     return report_tall


class Cellprofiler(pd.DataFrame):
    def __init__(
        self,
        *args,
        data_folder=None,
        nuclei_path="Secondary.csv",
        image_path=None,
        organoid_path=None,
        object_headers=["ImageNumber", "ObjectNumber"],
        filename_headers=["Date", "Drug", "Cell", "Replicate", "Conc /uM"],
        # regex_pattern=r"[\/\\](?P<Date>[\d]+)_.+_(?P<Cell>ISO[\d]+)_(?P<Drug>[A-Za-z0-9]+)_(?P<Concentration>[\d\w_-]+uM)(?:.+Position_(?P<Position>[\d]))?",
        regex_pattern=r"(?P<Date>([\d]{4}\-[\d]{2}\-[\d]{2})).+_(?P<Cell>ISO[\d]+)_(?P<Drug>[A-Za-z0-9]+)_(?P<Concentration>[\d\w_-]+uM)(?:.+Position_(?P<Position>[\d]))?",
        SAVE_FIG=False,
        SAVE_CSV=False,
        bad_cols_file="bad_cols.json",
        CHANNELS=1,
        ZEROISCONTROL=True,
        **kwargs,
    ):
        # sns.set()

        np.random.seed(42)
        # super(Cellprofiler,  self).__init__()
        # self.df = pd.read_csv(nuclei_file_path)
        # print(data_folder)
        attrs = {
            "data_folder": data_folder,
            "nuclei_path": nuclei_path,
            "image_path": image_path,
            "organoid_path": organoid_path,
            "SAVE_FIG": SAVE_FIG,
            "SAVE_CSV": SAVE_CSV,
            "bad_cols_file": bad_cols_file,
            "CHANNELS": CHANNELS,
            "ZEROISCONTROL": ZEROISCONTROL,
            "object_headers": object_headers,
            "filename_headers": filename_headers,
            "regex_pattern": regex_pattern,
            "index_headers": object_headers + filename_headers,
        }
        super(Cellprofiler, self).__init__(*args)
        self.attrs = attrs

        # print(self.attrs)
        # df = self.get_df()
        # super(Cellprofiler, self).__init__(*args,**kwargs)
        # return CellprofilerActive
        # super(Cellprofiler, self).__init__(,*args, **kwargs)
        # self = pd.DataFrame(df)
        # self.attrs.update(vars)
        # self.update(df)

        # print(df)
        # self.index_headers = self.object_headers + self.filename_headers
        # df = self.get_df()
        #     # .pipe(self.extract_headers,regex_pattern,filename_headers,index_headers)
        #     # .pipe(self.drop_bad_cols,bad_cols_file)
        #     # .pipe(self.zero_is_control)
        #     )

        # self.update(df.to_dict())
        # print(df)

        # print(self)

    def get_data(self):
        # df = self.get_df(self.attrs["data_folder"],self.attrs["nuclei_path"])

        df = (
            self.get_df(self.attrs["data_folder"], self.attrs["nuclei_path"])
            .pipe(
                self.extract_headers,
                self.attrs["regex_pattern"],
                self.attrs["filename_headers"],
                self.attrs["index_headers"],
            )
            .pipe(drop_bad_cols, self.attrs["bad_cols_file"])
            .pipe(zero_is_control)
            .pipe(drop_bad_index)
            .pipe(force_numeric)
        )
        # df = CellprofilerDataFrame(df,attrs=self.attrs)
        df
        # df = CellprofilerDataFrame(df)
        df.attrs.update(self.attrs)
        # return CellprofilerDataFrame(df,attrs=self.attrs)
        return df

    @property
    def _constructor(self):
        # print(self.attrs)
        # print("hello")
        return Cellprofiler

    def get_df(self, data_folder, nuclei_path):
        # return pd.DataFrame()
        # raw_df = self.read_data()
        # extracted_data = self.extract_filenames(
        #     self.get_filenames(raw_df),
        #     regex_pattern,
        #     filename_headers)
        return self.read_data(data_folder, nuclei_path)

    def read_data(self, data_folder, nuclei_path):
        nuclei_df = pd.read_csv(os.path.join(data_folder, nuclei_path))
        # if image_path is not None:
        #     image_df = pd.read_csv(f"{data_folder}/{image_path}")
        #     image_nuclei_df = pd.merge(
        #         nuclei_df,
        #         image_df,
        #         on="ImageNumber",
        #         how="left",
        #         # left_index = True,
        #         suffixes=("", "_Image"),
        #     )
        #     return image_nuclei_df
        return nuclei_df

    # %% Utils

    def extract_headers(self, df, regex_pattern, filename_headers, index_headers):
        filenames = self.get_filenames(df)
        extracted_data = self.extract_filenames(
            filenames, regex_pattern, filename_headers
        )
        df_indexed = df.join(extracted_data).set_index(index_headers)
        return df_indexed.drop(
            ["Metadata_FileLocation", "PathName_image"], axis=1, errors="ignore"
        )

    def get_filenames(self, df):
        return df[
            df.columns.intersection(["Metadata_FileLocation", "PathName_image"])
        ].iloc[:, 0]

    def extract_filenames(self, filenames, regex_pattern, filename_headers, decimals=3):
        extracted_data = filenames.str.extract(regex_pattern)
        extracted_data["filenames"] = filenames
        extracted_data["Replicate"] = extracted_data["Position"].fillna(1)
        extracted_data["Conc /uM"] = pd.to_numeric(
            extracted_data["Concentration"]
            .str.replace("_", ".")
            .str.replace("-", ".")
            .str.replace("uM", "")
        ).round(decimals=decimals)
        return extracted_data[filename_headers]



def zero_is_control(df):
    index_headers = list(df.index.names)
    df = df.reset_index()
    # index = df.index.to_frame()
    df["Drug"] = df["Drug"].where(df["Conc /uM"] != 0, other="Control")
    return df.set_index(index_headers)


def force_numeric(df):
    return (
        df.select_dtypes("number")
        .apply(pd.to_numeric)
        .select_dtypes("number")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=1)
        # .replace([np.inf, -np.inf], np.nan)
        # .dropna(axis=1)
        .sample(frac=1)
    )


def nuclei_primary_filter(df):
    return df.filter(regex=r"^((?!NucleiObjectsPrimary).)*$")


def drop_bad_cols(df, bad_cols_file):
    with open(bad_cols_file, "r") as f:
        # print(*raw_df_indexed.columns,sep='\n')
        bad_cols = json.load(f)
        return df.drop(bad_cols, axis=1, errors="ignore")


def drop_bad_index(df):
    return df.reindex(df.index.dropna())


# def balance_dataset(df,variable="Cell"):
#     g = df.groupby(level=variable,group_keys=False)
#     df = g.apply(lambda x: x.sample(g.size().min()))
#     print(df.index.get_level_values(variable).value_counts())
#     return df
