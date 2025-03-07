import sys
import pandas as pd
from pandas import DataFrame
import pickle
import numpy as np
from  describe import load
import sklearn


def load_thetas(s: str) -> dict:
    """
    Loads thetas for the predictions.
    :param s: path to thetas file.
    :return: dictionary of thetas.
    """
    try:
        with open(sys.argv[2], "rb") as f:
            weights = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file or directory: {s}")
    return weights


def standardization(df: DataFrame, titles: list) -> np.ndarray:
    """
    Normalize the data using standardization normalization.
    :param df: DataFrame
    :param titles: list of features
    :return: NumPy array
    """
    X = df[titles].to_numpy()
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    return X


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid function
    :param x: linear regression output
    :return: NumPy array of the sigmoid function values
    """
    return 1 / (1 + np.exp(-x))


def ft_predict(df: DataFrame, thetas: dict, titles: list) -> DataFrame:
    """
    Predicts the houses of students.
    :param df: dataframe
    :param thetas: dictionary of thetas for each feature
    :param titles: list of features
    :return:
    """
    houses = list(thetas.keys())

    # Standardize data to match thetas
    X = standardization(df, titles)

    # Save results in a dict for each house
    results = {house: sigmoid(np.dot(X, thetas[house][1:]) +
                            thetas[house][0]) for house in houses}
    # fill df['Hogwarts House'] with house having highest result
    df['Hogwarts House'] = [max(results, key=lambda house:
    results[house][i]) for i in range(len(df))]

    # df.insert(0, "Index", df.index)
    return df


def clean_data(df: DataFrame, titles: list) -> None:
    """
    Clean the data by replacing nan values by mean.
    :param df: dataframe
    :param titles: list of features
    :return: cleaned dataframe
    """
    df[titles] = df[titles].fillna(df[titles].mean())


def main():
    try:
        assert len(sys.argv) == 3, ("Wrong number of arguments, usage : "
                "python3 logreg_train.py <test_file> <thetas_file>")
        df = load(sys.argv[1])
        titles = [
            'Divination',
            'Muggle Studies',
            'Ancient Runes',
            'Transfiguration',
            'Charms',
        ]
        # df = df.iloc[:, 6:]
        clean_data(df, titles)
        print(df)
        thetas = load_thetas(sys.argv[2])
        df_predict = ft_predict(df, thetas, titles)
        df_predict.to_csv("houses_st.csv", columns=["Index", "Hogwarts House"],
                  index=False)
        print("\nPredictions done and saved in houses_st.csv\n")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
