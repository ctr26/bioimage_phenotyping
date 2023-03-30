import pandas as pd
import shap

# TODO, typee hinting
def get_shap_df(shaps):
    return (
        pd.DataFrame(shaps.values, columns=shaps.feature_names)
        .rename_axis("Sample")
        .reset_index()
        .melt(id_vars="Sample", var_name="Feature", value_name="Shap Value")
    )
    
def get_shap_values(
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
    df = self.df
    # X = np.array(df.apply(pd.to_numeric))
    X, y = df, list(df.index.get_level_values(variable))
    X100 = shap.utils.sample(np.array(X), 100)

    # y = df.index.get_level_values(variable)
    # y = DistanceMatrix().fit_transform(X)

    X_train, X_test, y_train, y_test = self.df.apply(
        pd.to_numeric
    ).bip.train_test_split(variable, groupby=groupby, augment=augment)
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