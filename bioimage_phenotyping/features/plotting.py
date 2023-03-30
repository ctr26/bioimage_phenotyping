import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib

def df_to_fingerprints(df, median_height=5, index_by="Drug",fig_size=(5,3)):
        # DRUGS = list(df.index.levels[3])
        LABELS = list(set(df.index.dropna().get_level_values(index_by).sort_values()))
        LABELS.sort()
        plt.rcParams["axes.grid"] = False
        fig, axes = plt.subplots(nrows=len(LABELS) * 2, figsize=fig_size, dpi=150)
        upper = np.mean(df.values.flatten()) + 1 * np.std(df.values.flatten())
        lower = np.mean(df.values.flatten()) - 1 * np.std(df.values.flatten())

        for i, ax in enumerate(axes.flat):
            drug = LABELS[int(np.floor(i / 2))]
            image = df.xs(drug, level=index_by)
            finger_print = image.median(axis=0)
            finger_print_image = matlib.repmat(finger_print.values, median_height, 1)

            if i & 1:
                # im = ax.imshow(image, vmin=image.min().min(),
                #                vmax=image.max().max(),cmap='Spectral')
                im = ax.imshow(
                    finger_print_image,
                    vmin=lower,
                    vmax=upper,
                    cmap="Spectral",
                    interpolation="nearest",
                )
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                im = ax.imshow(image, vmin=lower, vmax=upper, cmap="Spectral")
                ax.title.set_text(drug)
                # sns.heatmap(drug_df.values,ax=ax)
                ax.set(adjustable="box", aspect="auto", autoscale_on=False)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        
        fig.subplots_adjust(right=0.8)
        fig.colorbar(im, ax=axes.ravel().tolist())
        # fig.colorbar(im, cax=cbar_ax)
        
def df_to_fingerprints_facet(*args, **kwargs):
    upper = kwargs["vmax"]
    lower = kwargs["vmin"]
    data = kwargs.pop("data")
    data = data.drop([*args[2:]], 1)

    image = data.dropna(axis=1, how="all")

    rows, cols = image.shape
    median_height = 0.1
    gap_height = 0.15
    # median_rows = int(rows*median_height/100)
    image_rows_percent = 1 - (gap_height + median_height)
    one_percent = rows / image_rows_percent
    # print(one_percent,rows)
    gap_rows = int(gap_height * one_percent)
    median_rows = int(median_height * one_percent)

    # median_rows,gaps_rows=(rows,rows)
    finger_print = image.median(axis=0)

    finger_print_image = np.matlib.repmat(finger_print.values, median_rows, 1)
    all_data = np.vstack([image, np.full([gap_rows, cols], np.nan), finger_print_image])

    # fig,ax = plt.subplots(figsize=(5,3), dpi=150)
    # fig, ax = plt.figure(figsize=(5,3), dpi=150)
    # plt.figure()
    plt.imshow(all_data, vmin=lower, vmax=upper, cmap="Spectral")
    # sns.heatmap(all_data,vmin=lower, vmax=upper, cmap="Spectral",interpolation='nearest')
    fig, ax = (plt.gcf(), plt.gca())
    ax.set(adjustable="box", aspect="auto", autoscale_on=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_facecolor("white")
    ax.grid(False)
    # fig.add_subplot(2,2,1)
