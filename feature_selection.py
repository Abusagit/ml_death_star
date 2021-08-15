from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class MI_Scores:
    def __init__(self, X, y, engine, **kwargs):
        self.mi_scores = self._make_mi_scores(X, y, engine, **kwargs)

    @staticmethod
    def _make_mi_scores(X, y, engine, **kwargs):
        X = X.copy()
        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
        features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        
        mi = pd.Series(engine(X, y, features, **kwargs), index=X.columns, name="MI Scores").sort_values(ascending=True)
        return mi

    def plot_scores(self, **kwargs):
        width = np.arange(len(self.mi_scores))
        ticks = list(self.mi_scores.index)
        plt.figure(**kwargs)
        plt.barh(width, self.mi_scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")
        plt.show()


# TODO - mutual info classif
class MIScoresCategorical(MI_Scores):
    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, engine=mutual_info_classif, **kwargs)


class MIScoresRealVal(MI_Scores):
    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, engine=mutual_info_regression, **kwargs)


if __name__ == '__main__':
    a = mutual_info_classif()
