import pandas as pd
from pandas import DataFrame
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  #Agg backend for non-interactive use
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pickle


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid function
    :param x: linear regression output
    :return: NumPy array of the sigmoid function values
    """
    return 1 / (1 + np.exp(-x))


def standardization(df: DataFrame, titles: list) -> np.ndarray:
    """
    Normalize the data using standardization normalization.
    :param df: DataFrame
    :param titles: list of features
    :return: NumPy array
    """
    X = df[titles].to_numpy()
    Xmean = X.mean(axis=0)
    Xstd = X.std(axis=0)
    X = (X - Xmean) / Xstd
    return X


def model_accuracy(X: np.ndarray, thetas: dict, col: pd.Series) -> float:
    """
    Computes the accuracy of the model.
    :param X: NumPy array
    :param thetas: np.ndarray of theta for the features + intercept
    :param col: pandas Series
    :return: float
    """
    houses = col.unique()
    acc = np.max([sigmoid(np.dot(X, thetas[house][1:]) + thetas[house][0]) for
                 house in houses])
    print("Model accuracy: ", "{:.3f}".format(acc))


def train_logreg(df: DataFrame, titles: list) -> None:
    """
    Train a logreg model
    :param df: DataFrame
    :param titles: list of feature names
    :return: None
    """
    lr = 0.025
    n_iters = 500
    thetas = {}
    costs = {}

    # Normalize data and add first col of 1 for intercept
    X = standardization(df, titles)
    print(f"X values: {X}")

    houses = df['Hogwarts House'].unique()

    # Perform oneVSall model for each class
    for house in tqdm(houses):
        print("Training logreg for class", house)
        time.sleep(1)

        # Assigning 1 for curr house, 0 for the rest
        y = np.empty(len(X), dtype=int)
        for i in range(len(df['Hogwarts House'])):
            if df['Hogwarts House'].iloc[i] == house:
                y[i] = 1
            else:
                y[i] = 0

        theta = np.zeros(X.shape[1]) #1 theta per feature
        intercept = 0
        cost = []

        # Gradient descent
        for iter in range(n_iters):
            y_pred = sigmoid(np.dot(X, theta) + intercept)

            dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
            db = (1 / len(X)) * np.sum(y_pred - y)

            theta -= lr * dw
            intercept -= lr * db

            loss = (-1 / len(X)) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            cost.append(loss)

            if iter % 100 == 0:
                #current_accuracy = accuracy(sigmoid(np.dot(X, theta) +
                #                                    intercept) >= 0.7, y)
                print("iter:", iter, "cost:", "{:.2f}".format(loss))

        thetas[house] = theta
        # Add intercept to theta values
        thetas[house] = np.insert(thetas[house], 0, intercept)

        # Save and plot the cost function current house
        costs[house] = cost
        plt.plot(costs[house])

    plt.legend(houses)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.title('Logistic regression cost')
    plt.savefig("cost.png")

    # Computes the accuracy of the logreg
    model_accuracy(X, thetas, df['Hogwarts House'])

    # save thetas into file
    try:
        with open("thetas.pkl", "wb") as f:
            pickle.dump(thetas, f)
    except Exception:
        raise Exception("Could not save thetas.pkl")


def clean_data(df: DataFrame, titles: list) -> None:
    """
    Clean the data by replacing nan values by mean.
    :param df: dataframe
    :param titles: list of features
    :return: cleaned dataframe
    """
    #df[titles] = df[titles].ffill().bfill() #or this one
    df[titles] = df[titles].fillna(df[titles].mean())


def main():
    try:
        assert len(sys.argv) == 2, ("Wrong number of arguments, usage : "
                                    "python3 logreg_train.py <path-to-data>")
        df = load(sys.argv[1])
        titles = [
            'Astronomy',
            'Herbology',
            'Divination',
            'Muggle Studies',
            'Ancient Runes',
            'History of Magic',
            'Transfiguration',
            'Charms',
            'Flying'
        ]
        #df.dropna(inplace=True)
        clean_data(df, titles) # filling the nan values with mean or drop them??
        train_logreg(df, titles)

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
