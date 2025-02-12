import pandas as pd
from pandas import DataFrame
import sys
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive use
import matplotlib.pyplot as plt
from describe import load


def plot_scatter(df: DataFrame, titles: list) -> None:
    """
    Plots histograms for each feature in the dataset.
    :param df: Dataframe
    :param titles: list
    :return: None
    """
    sns.set_style("darkgrid")

    sns.pairplot(df, hue="Hogwarts House", vars=df[titles])

    plt.savefig("pair_plot.png")


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

        print("\nAccording to graph visualization, Arithmancy, "
              "Care of Magical creatures and Potions should be avoided features "
              "since they have similar distribution for the different houses.\n"
              "The best features for the logistic regression are:\n"
              "Divination\nMuggle Studies\nHistory of Magic\nCharms\nFlying")

    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()