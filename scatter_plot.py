import pandas as pd
from pandas import DataFrame
import sys
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive use
import matplotlib.pyplot as plt


def load(path: str) -> DataFrame | None:
    """
    takes a path as argument and returns the data set.
    :param path: str
    :return: DataFrame or None
    """
    try:
        file = pd.read_csv(path)
        ext = path.split(".")
        assert ext[len(ext) - 1].upper() == "CSV", "Wrong file format"

        file = pd.DataFrame(file)
        print("Loading dataset of dimensions", file.shape)
        return file
    except FileNotFoundError:
        raise FileNotFoundError(path)


def plot_scatter(df: DataFrame, titles: list) -> None:
    """
    Plots histograms for each feature in the dataset.
    :param df: Dataframe
    :param titles: list
    :return: None
    """
    palette = sns.color_palette("dark")
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
                curr_title = titles[i * 4 + j]
                ax[i, j].scatter(x=group.get_group('Ravenclaw')['Index'],
                                 y=group.get_group('Ravenclaw')[curr_title],
                                 label='Ravenclaw', color = palette[0], alpha = 0.5)
                ax[i, j].scatter(x=group.get_group('Slytherin')['Index'],
                                 y=group.get_group('Slytherin')[curr_title],
                                 label='Slytherin',
                                 color = palette[2], alpha = 0.5)
                ax[i, j].scatter(x=group.get_group('Gryffindor')['Index'],
                                 y=group.get_group('Gryffindor')[curr_title],
                                 label='Gryffindor',
                                 color = palette[3], alpha = 0.5)
                ax[i, j].scatter(x=group.get_group('Hufflepuff')['Index'],
                                 y=group.get_group('Hufflepuff')[curr_title],
                                 label='Hufflepuff',
                                 color = palette[8], alpha = 0.5)
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

    plt.savefig("scatter.png")


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
        plot_scatter(df, titles)
        print("According to graph vizualization, History of Magic and "
              "Transfiguration are the two similar features.")

    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()