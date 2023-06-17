# %%

import os
import pathlib
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Set the default show behavior to non-blocking
mpl.rcParams["backend"] = "TkAgg"
mpl.rcParams["interactive"] = True


import numpy as np
import pandas as pd
import seaborn as sns

# import bioimage_phenotyping as bip
from bioimage_phenotyping import Cellprofiler, features

# subprocess.run("make get.cellesce.data", shell=True)
sns.set()


VARIABLES = ["Conc /uM", "Date", "Drug"]
SHOW_PLOTS = False
SAVE_FIG = False
SAVE_CSV = True

kwargs_cellprofiler = {
    "data_folder": "data/results/unet",
    "nuclei_path": "all_FilteredNuclei.csv",
}

kwargs = kwargs_cellprofiler


def save_csv(df, path):
    df.to_csv(metadata(path))
    return df


results_folder = f'{kwargs["data_folder"]}/results'
pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)


def metadata(x):
    path = pathlib.Path(results_folder, x)
    print(path)
    return path


# %%


df = Cellprofiler(**kwargs).get_data().bip.clean().bip.preprocess()


df = pd.concat(
    [
        df.assign(**{"Population type": "Nuclei"}),
        df.bip.grouped_median("ObjectNumber")
        .assign(**{"Population type": "Organoid", "ObjectNumber": 0})
        .set_index("ObjectNumber", append=True)
        .reorder_levels(order=df.index.names),
    ]
).set_index("Population type", append=True)

print(
    f'Organoids: {df.xs("Organoid",level="Population type").bip.simple_counts()}',
    f'Nuclei: {df.xs("Nuclei",level="Population type").bip.simple_counts()}',
)

# %%

info = "finger_prints"
features.plotting.df_to_fingerprints(
    df.xs("Nuclei", level="Population type"), index_by="Drug", median_height=1
)

if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(info)
# %%

upper = np.nanmean(df.values.flatten()) + 3 * np.nanstd(df.values.flatten())
lower = np.nanmean(df.values.flatten()) - 3 * np.nanstd(df.values.flatten())

# %% Cell fingerprints

info = "fingerprints_cells"

g = sns.FacetGrid(
    df.reset_index(level="Cell"),
    # row="Drug",
    col="Cell",
    # height=2,
    aspect=1.61,
    sharey=False,
    sharex=False,
    height=3,
    # col_wrap=2,
)
cax = g.fig.add_axes([1.015, 0.13, 0.015, 0.8])
g.map_dataframe(
    features.plotting.df_to_fingerprints_facet,
    # "Drug",
    "Nuclei",
    "Drug",
    "Cell",
    cmap="Spectral",
    cbar=True,
    vmax=upper,
    vmin=lower,
)
plt.tight_layout()
plt.colorbar(cax=cax)
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"), bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(info)
# %% Cell and drug fingerprints


info = "fingerprints_drugs"

sns.set()
g = sns.FacetGrid(
    df.reset_index(level=["Cell", "Drug"]),
    row="Drug",
    col="Cell",
    # height=2,
    aspect=1.61,
    sharey=False,
    sharex=False,
    height=3,
    # col_wrap=2,
)
cax = g.fig.add_axes([1.015, 0.13, 0.015, 0.8])
g.map_dataframe(
    features.plotting.df_to_fingerprints_facet,
    "Drug",
    "Nuclei",
    "Drug",
    "Cell",
    cmap="Spectral",
    cbar=True,
    vmax=upper,
    vmin=lower,
)
plt.tight_layout()
plt.colorbar(cax=cax)
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"), bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(info)


# %%

info = "importance_median_control_points"
plt.figure(figsize=(3, 50))
sns.barplot(
    y="Feature",
    x="Importance",
    data=(
        df.xs("Organoid", level="Population type")
        .bip.feature_importance(variable="Cell")
        .reset_index()
        .pipe(save_csv, f"{info}.csv")
    ),
)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(info)
# %%
# sns.barplot(
#     y="Feature", x="Cumulative Importance",
#     data=df.bip.feature_importances(variable="Cell").reset_index()
# )
# plt.tight_layout()


def scoring(df, variable="Cell", augment=None, kfolds=5):
    return df.dropna(axis=1).bip.get_scoring_df(
        variable=variable, kfolds=kfolds, augment=augment
    )


# %%

# Drug scoring
info = "cell_predictions_image_vs_nuclei"
plot = sns.catplot(
    x="Kind",
    y="Score",
    col="Metric",
    # row="Cell",
    errorbar=None,
    # hue="Population type",
    data=(
        df.xs("Organoid", level="Population type")
        .bip.get_scoring_df(variable="Drug")
        # .pipe(bip.get_scoring_df, variable="Drug")
        .set_index("Metric")
        .loc[["f1-score", "recall", "precision"]]
        .reset_index()
        .pipe(save_csv, f"{info}.csv")
    ),
    sharey=False,
    kind="bar",
    col_wrap=2,
).set_xticklabels(rotation=45)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(info)
# %%
# %% Could do better with median per imagenumber
# plot = sns.catplot(
#     x="Kind",
#     y="Score",
#     col="Metric",
#     # row="Cell",
#     errorbar=None,
#     hue="Population type",
#     data=pd.concat(
#         [
#             (
#                 df.bip.get_score_report("Cell").assign(
#                     **{"Population type": "Nuclei"}
#                 )
#             ),
#             (
#                 df.bip.grouped_median("ObjectNumber")
#                 .bip.get_score_report("Cell")
#                 .assign(**{"Population type": "Organoid"})
#             ),
#         ])
#         .set_index("Metric")
#         .loc[['f1-score', 'recall','precision']]
#         .reset_index()
#         .pipe(save_csv,"Cell_predictions_image_vs_nuclei.csv"),
#     sharey=False,
#     kind="bar",
#     col_wrap=2,
# ).set_xticklabels(rotation=45)
# plt.tight_layout()
# if SAVE_FIG: plt.savefig(metadata("Cell_predictions_image_vs_nuclei.pdf"))
# if SHOW_PLOTS: plt.show()

# %%

info = "cell_predictions_organoid"
plot = sns.catplot(
    x="Kind",
    y="Score",
    # col="Metric",
    # row="Cell",
    errorbar=None,
    hue="Metric",
    data=(
        df.xs("Organoid", level="Population type")
        .bip.get_score_report("Cell")
        .assign(**{"Population type": "Organoid"})
        .set_index("Metric")
        .loc[["f1-score", "recall", "precision"]]
        .reset_index()
        .pipe(save_csv, f"{info}.csv")
    ),
    sharey=False,
    kind="bar",
    # col_wrap=3,
).set_xticklabels(rotation=45)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


print(info)
# %%

info = "Drug_predictions_per_organoid"
plot = sns.catplot(
    # x="Kind",
    y="Score",
    col="Metric",
    x="Cell",
    errorbar=None,
    hue="Kind",
    data=(
        df.bip.grouped_median("ObjectNumber")
        .groupby(level="Cell")
        .apply(lambda x: x.bip.get_score_report("Drug"))
        .reset_index()
    ),
    sharey=False,
    kind="bar",
    col_wrap=2,
).set_xticklabels(rotation=45)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(f"{info}")


# %%
info = "organoid_summary"
sns.catplot(
    y="Conc /uM",
    hue="Drug",
    x="Organoids",
    col="Cell",
    sharex=True,
    kind="bar",
    orient="h",
    errorbar=None,
    data=(
        df
        #   .grouped_median("ObjectNumber")
        .bip.groupby_counts("ImageNumber")
    ).reset_index(name="Organoids"),
)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(info)
# %%

info = "nuclei_summary"
sns.catplot(
    y="Conc /uM",
    hue="Drug",
    x="Nuclei",
    col="Cell",
    sharex=True,
    kind="bar",
    orient="h",
    errorbar=None,
    data=(df.bip.groupby_counts("ObjectNumber")).reset_index(name="Nuclei"),
)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(info)
# %%

info = "date_summary_organoids"
plot = sns.histplot(
    x="Date",
    weights="Organoids",
    hue="Drug",
    data=(
        df.bip.grouped_median("ObjectNumber")
        .bip.groupby_counts("ImageNumber")
        .reset_index(name="Organoids")
    ),
    multiple="stack",
)
plt.xticks(rotation=90)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(info)
# %%
info = "date_summary_organoid"
plot = sns.histplot(
    x="Date",
    weights="Organoids",
    hue="Drug",
    data=(
        df.bip.grouped_median("ObjectNumber")
        .bip.groupby_counts("ImageNumber")
        .reset_index(name="Organoids")
    ),
    multiple="stack",
)
plt.xticks(rotation=90)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()
print(info)
# %%
df_new = df.bip.select_features().bip.grouped_median("ObjectNumber")


# %%
info = "drug_predictions_per_organoid"
plot = sns.catplot(
    x="Kind",
    y="Score",
    col="Metric",
    # row="Cell",
    errorbar=None,
    # hue="Metric",
    data=(
        df.bip.grouped_median("ObjectNumber").bip.get_score_report("Drug").reset_index()
    ),
    sharey=False,
    kind="bar",
).set_xticklabels(rotation=45)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(info)


# %% Drug similarity
# A model is trained on each drug to predict either the the control drug or another drug, and the similarity is measured by the kl-divergence between the two models on a held out test set.

# So, you take all pairwise drugs, and train a model for each pair, the drug of interest is held out and the model is trained as a binary classification model.
# The similarity score is then either, the average probability of the


# model.predict(df_left_out.values)
# model.predict(df_left_out.values)


# def jensen_shannon_divergence(p, q):
#     # Calculate the average distribution
#     m = 0.5 * (p + q)
#     # Calculate the KL divergences from the average distribution to each input distribution
#     kl_p = np.sum(p * np.log2(p / m))
#     kl_q = np.sum(q * np.log2(q / m))
#     # Calculate the Jensen-Shannon divergence as the average of the two KL divergences
#     jsd = 0.5 * (kl_p + kl_q)
#     return jsd


# def similarity_score_kl(X_test, model, distance_fn=jensen_shannon_divergence):
#     # Use the predict_proba() method to get the predicted probability of each class for each sample in X_test
#     probas = model.predict_proba(X_test)

#     # Define the reference probabilities for the positive and negative classes
#     # Need epsilon to avoid log(0) errors
#     ref_p = [1 - 1e-9, 1e-9]  # equal probabilities for p and n
#     ref_n = [1e-9, 1 - 1e-9]  # equal probabilities for p and n

#     # Calculate the KL divergence between the predicted probabilities and the reference probabilities for each sample in X_test
#     divergences = []
#     for i in range(len(probas)):
#         div_p = distance_fn(probas[i], ref_p)
#         div_n = distance_fn(probas[i], ref_n)
#         divergences.append(kl_div_p - kl_div_n)

#     # Calculate the mean and standard deviation of the KL divergence scores
#     mean_div = np.mean(divergences)
#     std_div = np.std(divergences, ddof=1)

#     # Calculate the 95% confidence interval for the similarity score
#     n = len(divergences)
#     t_val = 2.0  # for a 95% confidence interval with n-1 degrees of freedom
#     conf_int = (
#         mean_div - t_val * std_div / np.sqrt(n),
#         mean_div + t_val * std_div / np.sqrt(n),
#     )

#     # Return the similarity score and its confidence interval
#     return mean_div, conf_int


# def pairwise_label_similarity(
#     df,
#     neg,
#     pos,
#     left_out,
#     level="Drug",
#     k_folds=5,
#     model=RandomForestClassifier(),
#     similarity_fun=similarity_score_kl,
# ):
#     labels = [neg, pos, left_out]

#     df_all_labels = df.bip.multikey_xs(
#         labels, level=level, drop_level=False
#     ).bip.balance_dataset(level)

#     df_left_out = df_all_labels.xs(left_out, level=level, drop_level=False)
#     # df_blind = df_all_labels.drop(left_out, level=level)

#     # Have to do a cross section rather than a drop
#     # incase any of pos==neg==left_out

#     df_blind = df_all_labels.bip.multikey_xs(
#     [neg, pos], level=level, drop_level=False
#     )

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


df.bip.pairwise_similarity(
    neg="Control", pos="G007", left_out="Dabrafenib", level="Drug", metric="js"
)

# pairwise_label_similarity(
#     df, neg="Control", pos="G007", left_out="Dabrafenib", level="Drug"
# )


def pairwise_matrix(level="Drug", metric="js",ci=None):
    labels = df.index.get_level_values(level).unique()

    index = pd.MultiIndex.from_product([labels, labels], names=["pos", "left_out"])
    # df_sim = pd.DataFrame(index=index)
    # TODO return ci or all divergences
    return pd.Series(index=index,name=metric).groupby(["pos", "left_out"]).apply(
        lambda group: (
            df.bip.pairwise_similarity(
                neg="Control",
                pos=group.name[0],
                left_out=group.name[1],
                level="Drug",
                metric="js",
                ci=ci,
            )[0]
        )
    )


df_sim = pd.DataFrame()

info = "drug_similarity"

# for metric in ["js", "kl"]:
#     df_sim[metric] = pairwise_matrix(level="Drug", metric=metric)

#     plot = sns.heatmap(
#         data=df_sim[metric].unstack(),
#     )
#     plot.ax.set_title(metric)

#     if SHOW_PLOTS:
#         plt.show()
#     else:
#         plt.close()




df_a =  pairwise_matrix(level="Drug", metric="js")

metrics = ["js", "kl"]
# df_sim = pd.DataFrame(columns=metrics)
df_sim.apply(lambda x: pd.Series([1,1]))

df_sim = pd.concat([pairwise_matrix(level="Drug", metric=metric,ci=None) for metric in metrics])

df_sim
# for metric in metrics:
df_sim[metric] = pairwise_matrix(level="Drug", metric=metric)

# df_sim = df_sim.stack().reset_index()
# df_sim.columns = ["pos", "left_out", "metric", "value"]

# def plot_heatmap(df_sim):
#     return sns.heatmap(df_sim.unstack())

# g = sns.FacetGrid(df_sim, col="metric", col_wrap=len(metrics))
# g = g.map_dataframe(
#     sns.heatmap,
#     x="pos",
#     y="left_out",
#     cbar=False,
#     annot=True,
#     cmap="coolwarm",
#     square=True,
# )
# g.fig.subplots_adjust(wspace=0.1, hspace=0.5)

# for ax, title in zip(g.axes.flat, metrics):
#     ax.set_title(title)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close()
# %% Concentration dependent study
info = "concentration_dependence"
#

plot = sns.catplot(
    x="Kind",
    y="Score",
    col="Metric",
    # row="Cell",
    errorbar=None,
    # hue="Metric",
    data=(
        df.bip.grouped_median("ObjectNumber").bip.get_score_report("Drug").reset_index()
    ),
    sharey=False,
    kind="bar",
).set_xticklabels(rotation=45)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# %%
info = "cell_predictions_image_vs_nuclei"
plot = sns.catplot(
    x="Kind",
    y="Score",
    col="Metric",
    row="Drug",
    errorbar=None,
    hue="Population type",
    data=(
        df
        # .xs("Nuclei",level="Population type")
        .groupby("Population type")
        .apply(scoring)
        .set_index("Metric")
        .loc[["f1-score", "recall", "precision"]]
        .reset_index()
        .pipe(save_csv, f"{info}.csv")
    ),
    sharey=False,
    kind="bar",
    col_wrap=2,
).set_xticklabels(rotation=45)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata(f"{info}.pdf"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

print(info)


# %% Section on predicting similarity between 0 and 1 conc
#
# %%


print("Exporting to notebook")
os.system("jupytext --to notebook notebooks/cellesce.py --update --execute")
