import pandas as pd
from pandas import DataFrame
import sys
import math


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


def ft_count(col: pd.Series) -> float:
    """
    Counts the number of values in a column.
    :param col: a pd.Series object
    :return: number of elements in the column
    """
    return len(col)


def ft_mean(col: pd.Series, count: float) -> float:
    """
    Computes the mean of a column.
    :param col: a pd.Series object
    :param count: float number of elements in the column
    :return: the mean of the column
    """
    return sum(col) / count


def ft_std(col: pd.Series, var: float) -> float:
    """
    Computes the standard deviation of a column.
    :param col: a pd.Series object
    :param var: float variance of the column
    :return: the standard deviation of the column
    """
    return pow(var, 0.5)


def ft_var(col: pd.Series, mean: float, count: float) -> float:
    """
    Computes the variance of a column.
    :param col: a pd.Series object
    :param mean: float mean of the column
    :param count: float number of elements in the column
    :return: the variance of the column
    """
    return sum((el - mean) ** 2 for el in col) / (count - 1)


def ft_min(col: pd.Series) -> float:
    """
    Searches for the minimum of a column and returns it.
    :param col: a pd.Series object
    :return: float
    """
    min = col[0]
    for el in col:
        if el < min:
            min = el
    return min


def ft_max(col: pd.Series) -> float:
    """
    Searches for the maximum of a column and returns it.
    :param col: a pd.Series object
    :return: float
    """
    max = col[0]
    for el in col:
        if el > max:
            max = el
    return max


def ft_percentiles(col: pd.Series, percentile: float, count: float) -> float:
    """
    Searches for the percentiles of a column and returns it.
    :param col: a pd.Series object
    :param percentile: desired percentile
    :param count: float number of elements in the column
    :return:
    """
    col = col.sort_values().reset_index(drop=True)

    per_pos = (count - 1) * percentile
    low_bound = math.floor(per_pos)
    high_bound = math.ceil(per_pos)

    # use linear interpolation like pd.describe()
    if low_bound == high_bound:
        return col[low_bound]
    else:
        return (col[low_bound] + (per_pos - low_bound) *
                (col[high_bound] - col[low_bound]))


def ft_skewness(col: pd.Series, mean: float, count: float, std:float) -> float:
    """
    Searches for the skewness of a column and returns it.
    Skewness = 0 when the distribution is normal.
    Skewness > 0 more weight is on the left side of the distribution.
    Skewness < 0 more weight is on the right side of the distribution.
    :param col: a pd.Series object
    :param mean: float mean of the column
    :param count: float number of elements in the column
    :param std: float standard deviation of the column
    :return: skewness of the column (float)
    """
    nominator = sum((el - mean) ** 3 for el in col) / count
    denominator = std ** 3
    return nominator / denominator


def ft_kurtosis(col: pd.Series, mean: float, count: float, std:float) -> float:
    """
    Searches for the kurtosis of a column and returns it.
    Subtracting 3 standardizes the result so that a normal
    distribution has a kurtosis of 0. Kurtosis < 0, distribution
    is flatter. Differences with the scipy library may arise
    due to bias approximations.
    :param col: a pd.Series object
    :param mean: float mean of the column
    :param count: float number of elements in the column
    :param std: float standard deviation of the column
    :return: kurtosis of the column (float)
    """
    nominator = sum((el - mean) ** 4 for el in col)
    denominator = std ** 4 * count
    return (nominator / denominator) - 3


def ft_describe(col: pd.Series, stats: DataFrame) -> None:
    """
    Calculates statistics about the given column series.
    :param col: a pd.Series object
    :param stats: statistics dataframe
    :return: None
    """
    col = col.dropna()

    count = ft_count(col)
    mean = ft_mean(col, count)
    var = ft_var(col, mean, count)
    std = ft_std(col, var)
    min = ft_min(col)
    max = ft_max(col)
    first_per = ft_percentiles(col, 0.25, count)
    second_per = ft_percentiles(col, 0.5, count)
    third_per = ft_percentiles(col, 0.75, count)
    skew = ft_skewness(col, mean, count, std)
    kurtosis = ft_kurtosis(col, mean, count, std)
    stats[col.name] = ["{:.6f}".format(count), "{:.6f}".format(mean),
                       "{:.6f}".format(std), "{:.6f}".format(min),
                        "{:.6f}".format(first_per), "{:.6f}".format(second_per),
                       "{:.6f}".format(third_per), "{:.6f}".format(max),
                      "{:.6f}".format(var), "{:.6f}".format(skew),
                       "{:.6f}".format(kurtosis)]


def main():
    try:
        assert len(sys.argv) == 2, "Wrong number of arguments"
        df = load(sys.argv[1])

        # Create an empty DataFrame for the statistics
        # Skewness is the measure of the asymmetry of the distribution of data.
        # Kurtosis is the measure of describing the distribution of data.
        stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%',
                 'max', 'var', 'skew', 'kurtosis']
        stats = pd.DataFrame(index=stats)

        for col in df.columns:
            if ((df.loc[:, col].dtype == 'int64' or
                df.loc[:, col].dtype == 'float64') and
                col != 'Index'):
                    ft_describe(df.loc[:, col], stats)
        print(stats)

    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()