import pandas as pd
from pandas import DataFrame
import sys
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive use
import matplotlib.pyplot as plt
from describe import load


def plot_histograms(df: DataFrame, titles: list) -> None:
    """
    Plots histograms for each feature in the dataset.
    :param df: Dataframe
    :param titles: list
    :return: None
    """
    palette = sns.color_palette("bright")
    sns.set_style("darkgrid")
    rows = 4
    cols = 4

    fig, ax = plt.subplots(rows, cols, tight_layout=True)
    fig.set_figheight(15)
    fig.set_figwidth(15)

    group = df.groupby("Hogwarts House")
    for i in range(rows):
        for j in range(cols):
            if i * 4 + j < len(titles):
                curr_col_title = titles[i * 4 + j]
                ax[i, j].hist(group.get_group('Ravenclaw')[curr_col_title], bins=40,
                              color=palette[0], label='Ravenclaw', alpha=0.5)
                ax[i, j].hist(group.get_group('Slytherin')[curr_col_title], bins=40,
                              color=palette[2], label='Slytherin', alpha=0.5)
                ax[i, j].hist(group.get_group('Gryffindor')[curr_col_title], bins=40,
                              color=palette[3], label='Gryffindor', alpha=0.5)
                ax[i, j].hist(group.get_group('Hufflepuff')[curr_col_title], bins=40,
                              color=palette[8], label='Hufflepuff', alpha=0.5)
                ax[i, j].set_title(titles[i * 4 + j])
                ax[i, j].set_ylabel('Frequency')
            else:
                # Hide unused subplots
                ax[i, j].axis('off')

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower right',
        bbox_to_anchor=(0.5, 0.1),
        ncol=1
    )

    plt.savefig("histogram.png")

def main():
    try:
        assert len(sys.argv) == 2, "Wrong number of arguments"
        df = load(sys.argv[1])
        titles = []
        for col in df.columns:
            if ((df.loc[:, col].dtype == 'int64' or
                df.loc[:, col].dtype == 'float64') and
                col != 'Index'):
                titles.append(col)

        plot_histograms(df, titles)
        print("According to graph vizualization, Arithmancy and "
              "Care of Magical creatures courses have a homogeneous "
              "score distribution between all four houses.")

    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()