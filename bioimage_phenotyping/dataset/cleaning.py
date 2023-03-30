from sklearn.ensemble import IsolationForest

# TODO make general
def model_inlier_detection(df,model=IsolationForest()):
    return df.loc[model.fit(df.values).predict(df) == 1]