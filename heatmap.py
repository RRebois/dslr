import pandas as pd
from pandas import DataFrame
import sys
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive use
import matplotlib.pyplot as plt
from describe import load


def ft_heatmap(df: DataFrame, titles: list) -> None:
    """
    Generates heatmap of correlation between columns.
    :param df: DataFrame
    :param titles: list of features
    :return: None
    """
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    sns.heatmap(df[titles].corr(), cmap="YlOrBr")
    plt.savefig("heatmap2.png")


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
        ft_heatmap(df, titles)

    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()