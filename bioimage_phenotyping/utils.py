
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from tqdm import tqdm

def pca_fun(df, pca=PCA(n_components=10)):
    return pd.DataFrame(pca.fit_transform(np.array(df)), index=df.index)
