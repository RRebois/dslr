import sys
import pandas as pd
from pandas import DataFrame, factorize


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


def main():
    try:
        assert len(sys.argv) == 2, "Wrong number of arguments"
        df = load(sys.argv[1])
        df["Hogwarts House"] = factorize(df["Hogwarts House"])[0]
        df.drop(columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"], inplace=True)
        result = df.corr()["Hogwarts House"].sort_values(ascending=False)
        print(result)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()