import pandas as pd


def calculate_pairwise_regression_score(group, neg, pos, level, metric, ci):
    """
    Calculate the pairwise regression score for a specific group.

    Parameters:
        group (pandas.DataFrame): The group data containing the regressor.
        neg (pandas.DataFrame): The negative group data.
        pos (pandas.DataFrame): The positive group data.
        level (str or int): The level of the regressor in the group index.
        metric (str): The metric to be used for calculating similarity.
        ci (float or None): The confidence interval for the score.

    Returns:
        float or tuple: The pairwise regression score. If ci is not None, returns a tuple containing the score and confidence interval.

    """
    regressor = group.index.get_level_values(level).unique().values[0]

    df_local = pd.concat([neg, group, pos])
    neg_label = neg.index.get_level_values(level).unique().values[0]
    pos_label = pos.index.get_level_values(level).unique().values[0]

    score = df_local.bip.pairwise_similarity(
        neg=neg_label, pos=pos_label, left_out=regressor, level=level, metric=metric, ci=ci
    )
    if ci is not None:
        score, ci = score
    return score


def calculate_regression_score(group, metric, level, ci=0.95):
    """
    Calculate the regression scores for each regressor in a group.

    Parameters:
        group (pandas.DataFrame): The group data.
        metric (str): The metric to be used for calculating similarity.
        level (str or int): The level of the regressor in the group index.
        ci (float): The confidence interval for the scores (default: 0.95).

    Returns:
        pandas.DataFrame: The regression scores for each regressor in the group.

    """
    regressors = group.index.get_level_values(level).unique()
    neg = group.xs(regressors.min(), level=level, drop_level=False)
    pos = group.xs(regressors.max(), level=level, drop_level=False)
    regressor_scores = group.groupby(level=level).apply(
        calculate_pairwise_regression_score,
        neg=neg,
        pos=pos,
        level=level,
        metric=metric,
        ci=ci,
    )
    return regressor_scores.apply(pd.Series).unstack()
