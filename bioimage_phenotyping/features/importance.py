import pandas as pd

def get_shap_df(shaps):
    return (
        pd.DataFrame(shaps.values, columns=shaps.feature_names)
        .rename_axis("Sample")
        .reset_index()
        .melt(id_vars="Sample", var_name="Feature", value_name="Shap Value")
    )