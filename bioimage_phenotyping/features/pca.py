from .. import utils
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
def fit_na(df,n_components=10):
    return PCA(n_components=n_components).fit(df.dropna(axis=1))


def fit_transform_na(df):
    pca_obj = fit_na(df)
    return df.dropna(axis=1).pipe(utils.pca_fun,pca_obj)



def explained_variance(df):
    pca_obj = fit_na(df)
    return pd.DataFrame(
            pca_obj.explained_variance_, columns=["Explained Variance"]
        ).assign(
            **{
                "Principal Component": np.arange(
                    0, len(pca_obj.explained_variance_)
                ),
            }
        )


def components(df):
    pca_obj = fit_na(df)
    return pd.DataFrame(
            pca_obj.components_, columns=["Components"]
        ).assign(
            **{
                "Principal Component": np.arange(
                    0, len(pca_obj.components_)
                ),
            }
        )
        

def components(df):
    df = df.dropna(axis=1)
    pca_obj = fit_na(df)
    return pd.DataFrame(pca_obj.components_, columns=df.columns).assign(
        **{
            "Principal Component": np.arange(0, len(pca_obj.components_)),
        }
    ).set_index("Principal Component")