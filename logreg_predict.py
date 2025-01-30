import sys
import pandas as pd
from pandas import DataFrame


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


def clean_data(df: DataFrame, titles: list) -> None:
    """
    Clean the data by replacing nan values by mean.
    :param df: dataframe
    :param titles: list of features
    :return: cleaned dataframe
    """
    #df[titles] = df[titles].ffill().bfill() #or this one
    df[titles] = df[titles].fillna(df[titles].mean())


def load_thetas(s: str) -> dict:



def main():
    try:
        assert len(sys.argv) == 3, ("Wrong number of arguments"
                "python3 logreg_train.py <path-to-data-test> <path-to-thetas>")
        df = load(sys.argv[1])
        titles = [
            #'Astronomy',
            #'Herbology',
            'Divination',
            'Muggle Studies',
            #'Ancient Runes',
            #'History of Magic',
            'Transfiguration',
            'Charms',
            #'Flying'
        ]
        #df.dropna(inplace=True)
        clean_data(df, titles) # filling the nan values with mean or drop them??
        thetas = load_thetas(sys.argv[2])
        train_logreg(df, titles)

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()