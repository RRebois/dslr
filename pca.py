import sys
from describe import load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    try:
        assert len(sys.argv) == 2, "Wrong number of arguments"
        df = load(sys.argv[1])
        # df["Hogwarts House"] = factorize(df["Hogwarts House"])[0]
        titles = []
        for col in df.columns:
            if ((df.loc[:, col].dtype == 'int64' or
                 df.loc[:, col].dtype == 'float64') and
                    col != 'Index'):
                df[col].fillna(df[col].mean(), inplace=True)
                titles.append(col)
            else:
                df.drop(col, axis=1, inplace=True)
        print(df)
        std_df = StandardScaler().fit_transform(df)
        pca = PCA(n_components=13)
        pca_df = pca.fit(std_df)
        explained_var_ratio = pca_df.explained_variance_ratio_ * 100
        pca_cumulate = explained_var_ratio.cumsum()

        print(f"Variances (percentage):\n", explained_var_ratio)
        print(f"Cumulative Variances:\n{pca_cumulate}")
        print(f"Number of components needed to explain 90% of the variance : "
              f"{len(pca_cumulate[pca_cumulate < 90]) + 1}")
        print(f"Number of components needed to explain 98% of the variance : "
            f"{len(pca_cumulate[pca_cumulate < 98]) + 1}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
