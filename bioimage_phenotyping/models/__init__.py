from sklearn import metrics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn import model_selection


def score_df_from_model(model, variable, X_test, y_test):
    scoring = X_test.apply(lambda x: model.predict([x])[0], axis=1).reset_index(
        variable, name="y_pred"
    )

    # scoring.groupby("Drug").
    #TODO Fix this
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


def get_score_report(
    df,
    variable="Cell",
    groupby=None,
    augment=None,
    model=RandomForestClassifier(),
):
    # labels, uniques = pd.factorize(df.reset_index()[variable])
    # X, y = df, list(df.index.get_level_values(variable))
    # uniques =
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    X_train, X_test, y_train, y_test = df.bip.train_test_split(
        variable, groupby=groupby, augment=augment
    )
    model.fit(X_train.values, y_train)
    return get_score_df_from_model(model, variable, X_test, y_test)



def df_to_training_data(df, variable):
    X, y = df, list(df.index.get_level_values(variable))
    return X, y


def groupby_train_split(df, variable, groupby, frac=0.8, seed=42):
    split_list = []
    for _, df_group in df.groupby(
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


def get_scoring_df(
    df,
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
                df.bip.get_score_report(df,
                    variable=variable, model=model, groupby=groupby, augment=augment
                ).assign(Fold=fold)
            )
            for fold in range(1, kfolds + 1)
        ]
    )



def train_test_split(
    df, variable="Cell", frac=0.8, augment=None, groupby=None, seed=42
):
    # df = self.df.sample(frac=1,random_state=seed)

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
        return groupby_train_split(
            df, variable, groupby, frac=0.8, seed=42, augment=augment
        )

    if augment is not None:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, stratify=y, random_state=seed
        )
        X_train, y_train = augment(X_train, y_train)
        return X_train, X_test, y_train, y_test

    return model_selection.train_test_split(X, y, stratify=y, random_state=seed)

    # train_idx, test_idx = next(gss.split(X, y, groups))

    # X_train = X.iloc[train_idx]
    # X_test = X.iloc[test_idx]

    # y_train = y.iloc[train_idx]
    # y_test = y.iloc[test_idx]

    # return X_train, X_test, y_train, y_test

    # X_train = (
    #     df.groupby(groupby, sort=False, as_index=False, group_keys=False)
    #     .sample(frac=1, random_state=seed)
    #     .sort_index()
    #     .sample(frac=0.8, random_state=seed)
    # )

    # if len(df) == len(X_train):
    #     X_train = df.sample(frac=frac, random_state=seed).sort_index()

    # dupe_df = pd.concat([df, X_train])
    # X_test = dupe_df[~dupe_df.index.duplicated(keep=False)]
    # y_train = X_train.index.to_frame()[[variable]].astype(str)
    # y_test = X_test.index.to_frame()[[variable]].astype(str)
    # # feature_df_in = feature_df_median_in
    # return X_train, X_test, y_train, y_test
