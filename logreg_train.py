import pandas as pd
from pandas import DataFrame
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pickle
from describe import load
matplotlib.use('Agg')  #Agg backend for non-interactive use


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


def model_accuracy(X: np.ndarray, thetas: dict, col: pd.Series):
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
    print("\nModel accuracy: ", "{:.3f}".format(acc))


def batch_gd(theta: np.ndarray, intercept: float, X: np.ndarray, y: np.ndarray, cost: list):
    """
    Batch gradient descent algorithm
    :param theta:
    :param intercept:
    :param X:
    :param y:
    :param cost:
    :return intercept, cost:
    """

    lr = 0.1
    n_iters = 1000
    for iter in range(n_iters):
        y_pred = sigmoid(np.dot(X, theta) + intercept)

        dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
        db = (1 / len(X)) * np.sum(y_pred - y)

        theta -= lr * dw
        intercept -= lr * db

        loss = (-1 / len(X)) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        cost.append(loss)

        if iter % 200 == 0:
            print("iter:", iter, "cost:", "{:.2f}".format(loss))

    return intercept, cost


def mini_batch_gd(theta: np.ndarray, intercept: float, X: np.ndarray, y: np.ndarray, cost: list):
    """
    Mini-batch gradient descent algorithm
    :param theta:
    :param intercept:
    :param X:
    :param y:
    :param cost:
    :return intercept, cost:
    """

    lr = 0.01
    n_iters = 200
    for iter in range(n_iters):
        batch_size = 50
        total_loss = 0
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            y_pred = sigmoid(np.dot(x_batch, theta) + intercept)

            dw = (1 / len(x_batch)) * np.dot(x_batch.T, (y_pred - y_batch))
            db = (1 / len(x_batch)) * np.sum(y_pred - y_batch)

            theta -= lr * dw
            intercept -= lr * db

            loss = (-1 / len(x_batch)) * np.sum(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))
            total_loss += loss
        avg_loss = total_loss / len(x_batch)
        cost.append(avg_loss)

        if iter % 50 == 0:
            print("iter:", iter, "cost:", "{:.2f}".format(avg_loss))

    return intercept, cost


def stochastic_gd(theta: np.ndarray, intercept: float, X: np.ndarray, y: np.ndarray, cost: list):
    """
    Stochastic gradient descent algorithm
    :param theta:
    :param intercept:
    :param X:
    :param y:
    :param cost:
    :return intercept, cost:
    """

    lr = 0.001
    n_iters = 200
    for iter in range(n_iters):
        total_loss = 0
        for i in range(len(X)):
            index = np.random.randint(0, len(X))
            x_i = X[index]
            y_i = y[index]

            y_pred = sigmoid(np.dot(x_i, theta) + intercept)

            dw = x_i * (y_pred - y_i)
            db = y_pred - y_i

            theta -= lr * dw
            intercept -= lr * db

            loss = -(y_i * np.log(y_pred) + (1 - y_i) * np.log(1 - y_pred))
            total_loss += loss
        avg_loss = total_loss / len(X)
        cost.append(avg_loss)

        if iter % 50 == 0:
            print("iter:", iter, "cost:", "{:.2f}".format(avg_loss))

    return intercept, cost


def train_logreg(df: DataFrame, titles: list, algo: str) -> None:
    """
    Train a logreg model
    :param df: DataFrame
    :param titles: list of feature names
    :param algo: str
    :return: None
    """
    thetas = {}
    costs = {}

    # Normalize data and add first col of 1 for intercept
    X = standardization(df, titles)

    houses = df['Hogwarts House'].unique()

    # Perform one VS all model for each class
    for house in tqdm(houses):
        print("\n\nTraining logreg for class", house)
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
        if algo == "batch":
            intercept, cost = batch_gd(theta, intercept, X, y, cost)
        if algo == "mini-batch":
            intercept, cost = mini_batch_gd(theta, intercept, X, y, cost)
        elif algo == "stochastic":
            intercept, cost = stochastic_gd(theta, intercept, X, y, cost)

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
    df[titles] = df[titles].fillna(df[titles].mean())


def main():
    try:
        assert len(sys.argv) == 3, ("Wrong number of arguments, usage : "
                                    "python3 logreg_train.py <path-to-data> <algorithm>")
        assert (sys.argv[2] == "batch" or sys.argv[2] == "stochastic"
                or sys.argv[2] == "mini-batch"), \
                "Wrong algorithm entered, must be 'batch', 'stochastic' or 'mini-batch'"
        algo = sys.argv[2]
        df = load(sys.argv[1])
        titles = [
            'Divination',
            'Muggle Studies',
            'Ancient Runes',
            'Transfiguration',
            'Charms',
        ]
        clean_data(df, titles)
        train_logreg(df, titles, algo)

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
