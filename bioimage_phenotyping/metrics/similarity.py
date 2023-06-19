from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
import numpy as np
from scipy.stats import t
import pandas as pd
from .. import models
from ..dataset import utils

def pairwise(
    df,
    neg,
    pos,
    left_out,
    level="Drug",
    k_folds=5,
    model=RandomForestClassifier(),
    metric="js",
    ci=None,
):
    labels = [neg, pos, left_out]

    df_all_labels = df.pipe(
        utils.multikey_xs, labels, level=level, drop_level=False
    ).pipe(utils.balance_dataset, level)

    df_left_out = df_all_labels.xs(left_out, level=level, drop_level=False)

    df_blind = utils.multikey_xs(df, [neg, pos], level=level, drop_level=False)
    if df_blind.empty:
        if ci is None:
            return pd.NA
        if ci is not None:
            return pd.NA, pd.NA
    
    X_train, X_test, y_train, y_test = df_blind.pipe(
        models.train_model,
        model,
        variable=level,
        frac=0.8,
        groupby=level,
        augment=None,
    )
    # similarity, conf_int = 
    return similarity_score(df_left_out, model=model, metric=metric, ci = ci)


#     def get_similarity(self):
#         return self.similarity
#     def get_confidence_interval(self):
#         return self.conf_int


def jensen_shannon_divergence(p, q, axis=0):
    # Calculate the average distribution
    m = 0.5 * (p + q)
    # Calculate the KL divergences from the average distribution to each input distribution
    kl_p = np.sum(p * np.log2(p / m), axis=axis)
    kl_q = np.sum(q * np.log2(q / m), axis=axis)
    # Calculate the Jensen-Shannon divergence as the average of the two KL divergences
    jsd = 0.5 * (kl_p + kl_q)
    return jsd


def similarity_score(X_test, model, metric="kl", ci=None):
    if "kl":
        return score_kl(X_test, model,ci=ci)
    if "js":
        return score_js(X_test, model,ci=ci)
    if "euclid":
        return score_euclidean(X_test, model,ci=ci)
    if "average":
        return score_average_probability(X_test, model, ci=ci)


def score_average_probability(X_test, model,ci=None):
    dist = np.mean(model.predict_proba(X_test), axis=0, keepdims=True)
    return distribution_to_ci(dist,ci=ci)


def score_euclidean(X_test, model,ci=None):
    pass


def score_js(X_test, model,ci=None):
    return(score_divergence(X_test, model, metric=entropy,ci=ci))

def score_kl(X_test, model,ci=None):
    return(score_divergence(X_test, model, metric=jensen_shannon_divergence,ci=ci))

def ci_to_t(ci, df):
    """
    Converts a confidence interval to a t-value using the t-distribution.
    
    Args:
    - ci: float, the confidence interval as a decimal (e.g., 0.95 for a 95% confidence interval).
    - df: int, the degrees of freedom.
    
    Returns:
    - float, the t-value corresponding to the given confidence interval and degrees of freedom.
    """
    alpha = 1 - ci
    return t.ppf(1 - alpha / 2, df)


def distribution_to_ci(distribution,ci=0.95,dof=10):
    t_val = ci_to_t(ci,dof)
    # Calculate the mean and standard deviation of the KL divergence scores
    mean_div = np.mean(distribution)
    std_div = np.std(distribution, ddof=1)

    # Calculate the 95% confidence interval for the similarity score
    n = len(distribution)
    # t_val = 2.0  # for a 95% confidence interval with n-1 degrees of freedom
    conf_int = (
        mean_div - t_val * std_div / np.sqrt(n),
        mean_div + t_val * std_div / np.sqrt(n),
    )
    return mean_div, conf_int


def score_divergence(X_test, model, metric=jensen_shannon_divergence,ci=None):
    # Use the predict_proba() method to get the predicted probability of each class for each sample in X_test
    probas = model.predict_proba(X_test)

    # Define the reference probabilities for the positive and negative classes
    # Need epsilon to avoid log(0) errors
    ref_p = [1 - 1e-9, 1e-9]  # equal probabilities for p and n
    ref_n = [1e-9, 1 - 1e-9]  # equal probabilities for p and n

    # Calculate the KL divergence between the predicted probabilities and the reference probabilities for each sample in X_test
    divergences = []
    # for i in range(len(probas)):
    div_p = metric(probas, ref_p, axis=1)
    div_n = metric(probas, ref_n, axis=1)
    divergences = div_p - div_n

    if ci is None:
        return divergences
    else:
        return distribution_to_ci(divergences,ci=ci)


    # # Calculate the mean and standard deviation of the KL divergence scores
    # mean_div = np.mean(divergences)
    # std_div = np.std(divergences, ddof=1)

    # # Calculate the 95% confidence interval for the similarity score
    # n = len(divergences)
    # t_val = 2.0  # for a 95% confidence interval with n-1 degrees of freedom
    # conf_int = (
    #     mean_div - t_val * std_div / np.sqrt(n),
    #     mean_div + t_val * std_div / np.sqrt(n),
    # )

    # # Return the similarity score and its confidence interval
    # return mean_div, conf_int


# def pairwise_label_similarity(
#     df,
#     neg,
#     pos,
#     left_out,
#     level="Drug",
#     k_folds=5,
#     model=RandomForestClassifier(),
#     similarity_fun=score_kl,
# ):
#     labels = [neg, pos, left_out]

#     df_all_labels = df.bip.multikey_xs(
#         labels, level=level, drop_level=False
#     ).bip.balance_dataset(level)

#     df_left_out = df_all_labels.xs(left_out, level=level, drop_level=False)
#     # df_blind = df_all_labels.drop(left_out, level=level)

#     # Have to do a cross section rather than a drop
#     # incase any of pos==neg==left_out

#     df_blind = df_all_labels.bip.multikey_xs([neg, pos], level=level, drop_level=False)

#     X_train, X_test, y_train, y_test = df_blind.bip.train_model(
#         model,
#         variable="Drug",
#         frac=0.8,
#         groupby="Drug",
#         augment=None,
#     )
#     similarity, conf_int = similarity_fun(df_left_out, model)
#     return similarity


# def similarity_metric_average_probability(X_test, model):
#     probas = model.predict_proba(X_test)
#     return np.mean(probas, axis=0, keepdims=True)


# #
# def similarity_metric_voting(kl_n, kl_p):
#     z = kl_n / kl_p - 1
#     return 1 / (1 + np.exp(-z))



# class PairwiseLabelSimilarity:
#     def __init__(
#         self,
#         neg,
#         pos,
#         left_out,
#         level="Drug",
#         k_folds=5,
#         model=RandomForestClassifier(),
#         similarity_metric="js",
#     ):
#         # self.neg = neg
#         # self.pos = pos
#         # self.left_out = left_out
#         self.level = level
#         self.k_folds = k_folds
#         self.model = model
#         self.similarity_fun = self.METRICS[similarity_metric]
