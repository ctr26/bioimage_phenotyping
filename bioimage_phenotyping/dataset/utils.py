import numpy as np
import pandas as pd
from typing import List, Any


from sklearn.decomposition import PCA

# from sklearn.model_selection import train_test_split


def drop_from_index(df, level):
    """Drops all but the specified index level from the dataframe

    Args:
        df: the dataframe
        level: the index level to keep

    Returns:
        A dataframe with all but the specified index level dropped

    """
    # levels = df.index.names
    # return df.droplevel([l for l in levels if l != level])
    return drop_from_list(list(df.index.names), level)


def keeplevel(df, level):
    return df.droplevel(drop_from_index(df, level))


def simple_counts(df):
    return df.count().iloc[0]


def grouped_median(df, group="ObjectNumber"):
    """Calculate the median of a column of a pandas DataFrame, grouped by a categorical variable.
    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame containing the data to be grouped and the column to calculate the median of.
    group : str
        The name of the categorical variable to group by. The default is "ObjectNumber".
    Returns
    -------
    pandas DataFrame
        A DataFrame containing the median value of the column and the number of rows in each group.
    """
    return df.groupby(level=drop_from_index(df, group)).median()


def drop_from_list(list_in, item):
    """Drops the item or items from the list_in list. Returns the list_in list with the item or items dropped."""
    item = [item] if isinstance(item, str) else item
    return list(set(list_in) - set(item))


def balance_dataset(df, variable):
    """Downsample to balance dataset.
    This function downsamples a dataset so that all classes are represented equally.
    It takes a dataframe and the variable to downsample on as input.
    It returns the downsampled dataframe.
    """
    g = df.groupby(level=variable, group_keys=False)
    df = g.apply(lambda x: x.sample(g.size().min()))
    # print(df.index.get_level_values(variable).value_counts())
    return df


# TODO fix or implement
def bootstrap(df, groups, size, group="ObjectNumber"):
    # boostrap_df = df.groupby(level=list(set(groups) - {group})).median()
    # random_sample, make n groups, median of each group
    return NotImplementedError(df, groups, size, group)


# 2DO fix
def groupby_not(df, groups, group):
    return df.groupby(level=list(set(groups) - {group}))


def groupby_conj(df, group):
    # Groupby bt NOT this group
    return df.groupby(level=drop_from_index(df, group), group_keys=False)


# With a multiindex group by everything except group and then coount
def groupby_counts(df, group):
    return groupby_conj(df, group).size()


def unique_levels(df, variable):
    """Return a Series of unique values in a variable of a MultiIndexed DataFrame.

    Args:
        df (DataFrame): The DataFrame to search.
        variable (str): The variable to search within.

    Returns:
        Series: A Series of unique values in the variable.

    """
    return df.index.get_level_values(variable).to_series().unique()


def pca_fun(df, pca=PCA(n_components=10)):
    return pd.DataFrame(pca.fit_transform(np.array(df)), index=df.index)


def multikey_xs(df, key: List[Any], level: str, *args, **kwargs):
    # return pd.concat(
    #     [
    #         group
    #         for label, group in df.groupby(key, level=level, *args, **kwargs)
    #         if label in key
    #     ]
    # )
    return pd.concat(
        [
            df.xs(label, level=level, *args, **kwargs)
            for label in key
            if label in df.index.get_level_values(level)
        ],
        axis=0,
    )
