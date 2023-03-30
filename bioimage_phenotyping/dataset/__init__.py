from .dataset import Cellprofiler
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("bip")
class BioImageDataFrame:
    def __init__(self, df):
        self.attrs = None
        self.df = df