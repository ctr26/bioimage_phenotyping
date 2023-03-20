# %%
# %%

import subprocess

subprocess.run("make get.cellesce.data", shell=True)

import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.pipeline import Pipeline

import bioimage_phenotyping as bip
from bioimage_phenotyping import Cellprofiler

sns.set()
from bioimage_phenotyping import Cellprofiler, features

VARIABLES = ["Conc /uM", "Date", "Drug"]
SAVE_FIG = True
SAVE_CSV = True

kwargs_cellprofiler = {
    "data_folder": "results/unet",
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

features.plotting.df_to_fingerprints(
    df.xs("Nuclei", level="Population type"), index_by="Drug", median_height=1
)

plt.savefig(metadata("finger_prints.pdf"))
plt.show()
plt.close()
print("OK")
# %%
upper = np.nanmean(df.values.flatten()) + 2 * np.nanstd(df.values.flatten())
lower = np.nanmean(df.values.flatten()) - 2 * np.nanstd(df.values.flatten())

# %% Cell fingerprints
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
plt.savefig(metadata("fingerprints_cells.pdf"), bbox_inches="tight")
plt.show()
plt.close()
print("OK")
print("OK")
# %% Cell and drug fingerprints

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
plt.savefig(metadata("fingerprints_drugs.pdf"), bbox_inches="tight")
plt.show()
plt.close()
print("OK")


# %%
plt.figure(figsize=(3, 50))
sns.barplot(
    y="Feature",
    x="Importance",
    data=(
        df.xs("Organoid", level="Population type")
        .bip.feature_importances(variable="Cell")
        .reset_index()
        .pipe(save_csv, "importance_median_control_points.csv")
    ),
)
plt.tight_layout()
plt.savefig(metadata("importance_median_control_points.pdf"))
plt.show()
print("OK")
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

plot = sns.catplot(
    x="Kind",
    y="Score",
    col="Metric",
    # row="Cell",
    ci=None,
    # hue="Population type",
    data=(
        df.xs("Organoid", level="Population type")
        .pipe(bip.dataset.get_scoring_df, variable="Drug")
        .set_index("Metric")
        .loc[["f1-score", "recall", "precision"]]
        .reset_index()
        .pipe(save_csv, "Cell_predictions_image_vs_nuclei.csv")
    ),
    sharey=False,
    kind="bar",
    col_wrap=2,
).set_xticklabels(rotation=45)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata("Cell_predictions_image_vs_nuclei.pdf"))
plt.show()
plt.close()
print("OK")
# %%
# %% Could do better with median per imagenumber
# plot = sns.catplot(
#     x="Kind",
#     y="Score",
#     col="Metric",
#     # row="Cell",
#     ci=None,
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
# plt.show()

# %%


plot = sns.catplot(
    x="Kind",
    y="Score",
    # col="Metric",
    # row="Cell",
    ci=None,
    hue="Metric",
    data=(
        df.xs("Organoid", level="Population type")
        .bip.get_score_report("Cell")
        .assign(**{"Population type": "Organoid"})
        .set_index("Metric")
        .loc[["f1-score", "recall", "precision"]]
        .reset_index()
        .pipe(save_csv, "Cell_predictions_organoid.csv")
    ),
    sharey=False,
    kind="bar",
    # col_wrap=3,
).set_xticklabels(rotation=45)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata("Cell_predictions_organoid.pdf"))
plt.show()
plt.close()
print("OK")
# %%
plot = sns.catplot(
    # x="Kind",
    y="Score",
    col="Metric",
    x="Cell",
    ci=None,
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
    plt.savefig(metadata("Drug_predictions_per_organoid.pdf"))
plt.show()
plt.close()
print("OK")

# %%
sns.catplot(
    y="Conc /uM",
    hue="Drug",
    x="Organoids",
    col="Cell",
    sharex=True,
    kind="bar",
    orient="h",
    ci=None,
    data=(
        df
        #   .grouped_median("ObjectNumber")
        .bip.groupby_counts("ImageNumber")
    ).reset_index(name="Organoids"),
)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata("Organoid_Summary.pdf"))
plt.show()
plt.close()
print("OK")
# %%
sns.catplot(
    y="Conc /uM",
    hue="Drug",
    x="Nuclei",
    col="Cell",
    sharex=True,
    kind="bar",
    orient="h",
    ci=None,
    data=(df.bip.groupby_counts("ObjectNumber")).reset_index(name="Nuclei"),
)
plt.tight_layout()
plt.savefig(metadata("Nuclei_Summary.pdf"))
plt.show()
plt.close()
print("OK")
# %%
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
plt.savefig(metadata("Date_summary_organoids.pdf"))
plt.show()
plt.close()
print("OK")
# %%

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
plt.savefig(metadata("Date_summary_organoids.pdf"))
plt.show
plt.close()
print("OK")
# %%
df_new = df.bip.select_features().bip.grouped_median("ObjectNumber")


# %%
plot = sns.catplot(
    x="Kind",
    y="Score",
    col="Metric",
    # row="Cell",
    ci=None,
    # hue="Metric",
    data=(
        df.bip.grouped_median("ObjectNumber").bip.get_score_report("Drug").reset_index()
    ),
    sharey=False,
    kind="bar",
).set_xticklabels(rotation=45)
plt.show()
plt.close()
print("OK")
# %%

plot = sns.catplot(
    x="Kind",
    y="Score",
    col="Metric",
    row="Drug",
    ci=None,
    hue="Population type",
    data=(
        df
        # .xs("Nuclei",level="Population type")
        .groupby("Population type")
        .apply(scoring)
        .set_index("Metric")
        .loc[["f1-score", "recall", "precision"]]
        .reset_index()
        .pipe(save_csv, "Cell_predictions_image_vs_nuclei.csv")
    ),
    sharey=False,
    kind="bar",
    col_wrap=2,
).set_xticklabels(rotation=45)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata("Cell_predictions_image_vs_nuclei.pdf"))
plt.show()
plt.close()
print("OK")
# %% Concentration dependent study
print()

#%% Section on predicting similarity between 0 and 1 conc

df
