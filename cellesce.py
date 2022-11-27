# %%
import subprocess
subprocess.run("make get.data",shell=True)
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import SelectFromModel, RFECV, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import model_selection
import pathlib
import random

from bioimage_phenotyping import Cellprofiler
sns.set()
from bioimage_phenotyping import Cellprofiler
from bioimage_phenotyping import features

VARIABLES = ["Conc /uM", "Date", "Drug"]
SAVE_FIG = True
SAVE_CSV = True

# data_folder = "analysed/210720 - ISO49+34 - projection_XY/unet_2022/project_XY_all"
# data_folder = "analysed/210720 - ISO49+34 - projection_XY/unet_2022/project_XY_objects"

# pd.read_csv("analysed/_2019_cellesce_unet_splineparameters_aligned/raw/projection_XY/Secondary.csv")

kwargs = {
    "data_folder": "analysed/210720 - ISO49+34 - projection_XY/unet_2022/project_XY_all",
    "nuclei_path": "object_filteredNuclei.csv", 
}
kwargs = {
    "data_folder": "analysed/210720 - ISO49+34 - projection_XY/unet_2022/project_XY_objects",
    "nuclei_path": "Secondary.csv",
}


kwargs={
    "data_folder": "analysed/_2019_cellesce_unet_splineparameters_aligned/raw/projection_XY/",
    "nuclei_path": "Secondary.csv"
}


kwargs={
    "data_folder": "analysed/_2019_cellesce_unet_splineparameters_aligned/raw/projection_XY/",
    "nuclei_path": "Secondary.csv"
}

kwargs_cellprofiler= {
    "data_folder": "old_results/analysed/210720 - ISO49+34 - projection_XY/unet_2022/project_XY_objects",
    "nuclei_path": "object_filteredNuclei.csv",
}


# kwargs_splinedist = {
#     "data_folder": "analysed/cellprofiler",
#     "nuclei_path": "objects_FilteredNuclei.csv",
# }


# kwargs_splinedist = {
#     "data_folder": "analysed/cellesce_splinedist_controlpoints",
#     "nuclei_path": "Secondary.csv",
# }

# kwargs_splinedist = {
#     "data_folder": "control_points",
#     "nuclei_path": "objects_FilteredNuclei.csv",
# }

# kwargs_cellprofiler = {
#     "data_folder": "analysed/cellprofiler/splinedist",
#     "nuclei_path": "objects_FilteredNuclei.csv",
# }


# kwargs_cellprofiler = {
#     "data_folder": "old_results/analysed/cellprofiler/splinedist_32",
#     "nuclei_path": "objects_FilteredNuclei.csv",
# }

kwargs_cellprofiler = {
    "data_folder": "results/unet",
    "nuclei_path": "objects_FilteredNuclei.csv",
}

kwargs_cellprofiler = {
    "data_folder": "old_results/analysed/cellprofiler/splinedist_32",
    "nuclei_path": "objects_FilteredNuclei.csv",
}

kwargs_cellprofiler = {
    "data_folder": "results/unet",
    "nuclei_path": "all_FilteredNuclei.csv",
}



kwargs = kwargs_cellprofiler

def save_csv(df,path):
    df.to_csv(metadata(path))
    return df


results_folder = f'{kwargs["data_folder"]}/results'
pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)

def metadata(x):
    path = pathlib.Path(results_folder,x)
    print(path)
    return path
# %%


df = (Cellprofiler(**kwargs)
      .get_data()
      .bip.clean()
      .bip.preprocess())

# rows,features = df.shape
# df = df.iloc[:,random.sample(range(0, features), 32)]


df = pd.concat(
        [
                df.assign(
                    **{"Population type": "Nuclei"}
                ),
                df
                .bip.grouped_median("ObjectNumber")
                .assign(**{"Population type": "Organoid","ObjectNumber":0})
                .set_index("ObjectNumber",append=True)
                .reorder_levels(order=df.index.names)
        ]).set_index("Population type",append=True)


print(
    f'Organoids: {df.xs("Organoid",level="Population type").bip.simple_counts()}',
    f'Nuclei: {df.xs("Nuclei",level="Population type").bip.simple_counts()}',
)

# %%

features.plotting.df_to_fingerprints(df
                    .xs("Nuclei",level="Population type"),
                   index_by="Drug",
                   median_height=1)

# df_to_fingerprints(df,index_by="Drug", median_height=1)
# plt.tight_layout()
plt.savefig(metadata("finger_prints.pdf"))
plt.show()
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

# %% Cell and drug finerprints

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

# %%
# importance = df.bip.feature_importances(variable="Cell").reset_index()

# sns.barplot(
#     y="Feature", x="Importance",
#     data=df.bip.feature_importances(variable="Cell").reset_index()
# )
# plt.show()
# %%
plt.figure(figsize=(3, 50))
sns.barplot(
    y="Feature", x="Importance",
    data=(df
          .xs("Organoid",level="Population type")
          .bip.feature_importances(variable="Cell")
          .reset_index()
          .pipe(save_csv,"importance_median_control_points.csv")
          )
)
plt.tight_layout()
plt.savefig(metadata("importance_median_control_points.pdf"))
plt.show()
# %%
# sns.barplot(
#     y="Feature", x="Cumulative Importance",
#     data=df.bip.feature_importances(variable="Cell").reset_index()
# )
# plt.tight_layout()   

def scoring(df, variable="Cell", augment=None, kfolds=5):
    return (
        df
        .dropna(axis=1)
        .bip.get_scoring_df(variable=variable, kfolds=kfolds, augment=augment)
    )


# %%

plot = sns.catplot(
    x="Kind",
    y="Score",
    col="Metric",
    # row="Cell",
    ci=None,
    hue="Population type",
    data=(df
        .groupby("Population type")
        .apply(scoring)
        .set_index("Metric")
        .loc[['f1-score', 'recall','precision']]
        .reset_index()
        .pipe(save_csv,"Cell_predictions_image_vs_nuclei.csv")),
    sharey=False,
    kind="bar",
    col_wrap=2,
).set_xticklabels(rotation=45)
plt.tight_layout()
if SAVE_FIG: plt.savefig(metadata("Cell_predictions_image_vs_nuclei.pdf"))
plt.show()
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
    data=(df.xs("Organoid",level="Population type"),
            .bip.get_score_report("Cell")
            .assign(**{"Population type": "Organoid"})
            .set_index("Metric")
            .loc[['f1-score', 'recall','precision']]
            .reset_index()
            .pipe(save_csv,"Cell_predictions_organoid.csv")
            ),
    sharey=False,
    kind="bar",
    # col_wrap=3,
).set_xticklabels(rotation=45)
plt.tight_layout()
if SAVE_FIG: plt.savefig(metadata("Cell_predictions_organoid.pdf"))
plt.show()

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
if SAVE_FIG: plt.savefig(metadata("Drug_predictions_per_organoid.pdf"))
plt.show()
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
if SAVE_FIG: plt.savefig(metadata("Organoid_Summary.pdf"))
plt.show()
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
    data=(df.bip.groupby_counts("ObjectNumber"))
    .reset_index(name="Nuclei"),
)
plt.tight_layout()
if SAVE_FIG: plt.savefig(metadata("Nuclei_Summary.pdf"))
plt.show()
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
if SAVE_FIG: plt.savefig(metadata("Date_summary_organoids.pdf"))
plt.show()
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
if SAVE_FIG: plt.savefig(metadata("Date_summary_organoids.pdf"))
plt.show()
# %%
# sns.clustermap(
#     (
#         df.bip.grouped_median("ObjectNumber")
#         .bip.keeplevel(["Cell", "Drug"])
#         .T.corr()
#     )
# )
# plt.show()
# sns.clustermap(
#     (df.bip.grouped_median("ObjectNumber")
#      .bip.keeplevel(["Cell"])
#      .T.corr())
# )
# plt.show()
# sns.clustermap(
#     (df.bip.grouped_median("ObjectNumber")
#      .bip.keeplevel(["Drug"])
#      .T.corr())
# )
# plt.show()


# # %%
# # model = Pipeline([
# #                     ("PCA", PCA()),
# #                     ("modelselect", SelectFromModel(RandomForestClassifier())),
# #                     ("RandomForest",RandomForestClassifier())
# #                   ])
# model = RandomForestClassifier()
# # model = Pipeline([
# #                     ("PCA", PCA()),
# #                     ("RandomForest",RandomForestClassifier())
# #                   ])

# importance = df.bip.feature_importances(RandomForestClassifier(), variable="Cell")
# print("Nuclei Cell")
# importance = df.bip.grouped_median().bip.feature_importances(
#     model, variable="Cell"
# )
# print("Organoid Cell")

# importance = df.bip.feature_importances(RandomForestClassifier(), variable="Drug")
# print("Nuclei Drug")

# importance = df.bip.grouped_median().bip.feature_importances(model, variable="Drug")
# print("Organoid Drug")



# # fig_dims = (8, 6)

# sns.barplot(
#     y="Feature", x="Cumlative importance",
#     data=df.bip.feature_importances(variable="Cell")
# )
# plt.tight_layout()


    
    
# %%
df_new = df.bip.select_features().bip.grouped_median("ObjectNumber")
# importance = df.classification_report(model)
# # %%
# VARS = ["Cell", "Drug", "Conc /uM"]
# variable = VARS[1]
# # for VAR in VARS:
# def get_score_report(df, variable="Cell"):
#     # labels, uniques = pd.factorize(df.reset_index()[variable])
#     X, y = df, list(df.index.get_level_values(variable))
#     uniques = df.index.get_level_values(variable).to_series().unique()
#     X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
#     X_train, X_test, y_train, y_test = df.bip.train_test_split(variable)
#     model.fit(X_train, y_train)
#     # y_pred = pd.Series(model.predict(X_test), index=X_test.index)

#     report = pd.DataFrame(
#         metrics.classification_report(y_test, model.predict(X_test), output_dict=True)
#     ).drop(["accuracy", "macro avg", "weighted avg"], axis=1)
#     report_tall = (
#         report.rename_axis("Metric")
#         .melt(
#             # id_vars="Metric",
#             var_name="Kind",
#             value_name="Score",
#             ignore_index=False,
#         )
#         .assign(**{"Variable": variable})
#         .reset_index()
#     )
#     report_tall["Cohen Kappa"] = metrics.cohen_kappa_score(
#         y_test, model.predict(X_test)
#     )
#     return report_tall


# %%
# data = report_df.set_index("Variable").xs("Drug")

# %%
plot = sns.catplot(
    x="Kind",
    y="Score",
    col="Metric",
    # row="Cell",
    ci=None,
    # hue="Metric",
    data=(
        df.bip.grouped_median("ObjectNumber").bip.get_score_report("Drug")
        .reset_index()
    ),
    sharey=False,
    kind="bar",
).set_xticklabels(rotation=45)
plt.show()

# %%
# data=pd.concat([
#         get_score_report(df.bip.grouped_median("ObjectNumber"),"Cell").assign(**{"Population type": "Organoid"}),
#         get_score_report(df,"Cell").assign(**{"Population type": "Nuclei"})
#         ]),

# df.bip.grouped_median("ObjectNumber"),"Cell").groupby(level="Cell").apply(get_score_report)
# plot = sns.catplot(
#     x="Kind",
#     y="Score",
#     col="Metric",
#     # row="Cell",
#     ci=None,
#     # hue="Metric",
#     data=report_df.xs("Drug", level="Variable").reset_index(),
#     sharey=False,
#     kind="bar",
# ).set_xticklabels(rotation=45)
# %%


# # a.bip.test_fun()

# a.groupby("col1").apply(lambda d: d.bip.test_fun())

# %%

# df.pipe(get_score_report,variable)
# (df
#  .groupby(level=["Conc /uM"])
#  .apply(get_score_report,)
# )

# report_df = [df.groupby(level=variable).apply(
#     lambda x: get_score_report(x,variable)
#     )
#  for variable in [VARS[1]]];report_df

# report_df = [df.groupby(level=variable).apply(
#     lambda x: get_score_report(x,variable)
#     )
#  for variable in [VARS[0]]];report_df

# report_df = [df.groupby(level=variable).apply(
#     lambda x: get_score_report(x,variable)
#     )
#  for variable in [VARS[2]]];report_df
# %%
# return report_tall
# a = get_score_report(df,"Cell")
# b = get_score_report(df,"Drug")
# c = get_score_report(df,"Conc /uM")
# report_df = pd.concat(
#     [
#         get_score_report(df.bip.grouped_median("ObjectNumber"), variable).assign(
#             **{"Population type": "Median"}
#         )
#         for variable in VARS
#     ]
# ).set_index(["Metric", "Kind", "Variable", "Population type"])

# report_df = pd.concat(
#         get_score_report(df.bip.grouped_median("ObjectNumber"), "Cell").assign(
#             **{"Population type": "Median"}
#         )
# ).set_index(["Metric", "Kind", "Variable", "Population type"])
# %%

# df.bip.grouped_median("ObjectNumber").apply()

# (pd.concat([
#     df.assign(**{"Population type": "Organoid"}),
#     df.bip.grouped_median("ObjectNumber")
#     ])
#     .apply(lambda x: get_score_report(x,"Cell"))
# )

# df.bip.grouped_median("ObjectNumber").apply(lambda x: get_score_report(x,"Cell"))

# pd.concat([
# get_score_report(df.bip.grouped_median("ObjectNumber"),"Cell").assign(**{"Population type": "Organoid"}),
# get_score_report(df,"Cell").assign(**{"Population type": "Nuclei"})
# ])
# get_score_report(df, "Cell").assign(**{"Population type": "Median"})
# get_score_report(df.bip.grouped_median("ObjectNumber"), "Cell")
