# %%

import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

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
