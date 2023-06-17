from sklearn import metrics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn import model_selection


def score_df_from_model(model, variable, X_test, y_test):
    scoring = X_test.apply(lambda x: model.predict([x])[0], axis=1).reset_index(
        variable, name="y_pred"
    )

    # scoring.groupby("Drug").
    # TODO Fix this
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


def train_model(
    df,
    model,
    variable="Cell",
    groupby=None,
    augment=None,
    frac=0.8
):
    X_train, X_test, y_train, y_test = train_test_split(
        df=df, variable=variable, groupby=groupby, augment=augment,frac=frac,
    )
    model.fit(X_train.values, y_train)

    return X_train, X_test, y_train, y_test


def get_score_report(
    df,
    variable="Cell",
    groupby=None,
    augment=None,
    model=RandomForestClassifier(),
    frac=0.8,
):
    X_train, X_test, y_train, y_test = train_model(
        df=df, variable="Cell", groupby=groupby, augment=augment, model=model
    )
    # labels, uniques = pd.factorize(df.reset_index()[variable])
    # X, y = df, list(df.index.get_level_values(variable))
    # uniques =
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        df=df, variable=variable, groupby=groupby, augment=augment, frac=frac
    )
    model.fit(X_train.values, y_train)
    return score_df_from_model(model, variable, X_test, y_test)


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
                get_score_report(
                    df, variable=variable, model=model, groupby=groupby, augment=augment
                ).assign(Fold=fold)
            )
            for fold in range(1, kfolds + 1)
        ]
    )


def train_test_split(
    df, variable="Cell", frac=0.8, augment=None, groupby=None, seed=42
):
    X = df
    y = df.index.to_frame()[[variable]].astype(str)

    if groupby is not None:
        return groupby_train_split(
            df,
            variable,
            groupby,
            frac=frac,
            seed=seed,
        )

    if augment is not None:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, stratify=y, random_state=seed
        )
        X_train, y_train = augment(X_train, y_train)
        return X_train, X_test, y_train, y_test

    return model_selection.train_test_split(X, y, stratify=y, random_state=seed)
