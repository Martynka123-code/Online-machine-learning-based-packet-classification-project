import pandas as pd

from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)

from sklearn.ensemble import (
    RandomForestClassifier
)


class CorrelationFilter(
    BaseEstimator,
    TransformerMixin
):
    """
    Removes highly correlated features.

    Keeps the more important feature
    based on RandomForest feature importance.
    """

    def __init__(
            self,
            threshold=0.85,
            random_state=32
    ):

        self.threshold = threshold
        self.random_state = random_state

        self.to_drop_ = []
        self.selected_features_ = None

    # ======================================================
    # FIT
    # ======================================================

    def fit(self, X, y=None):

        X = pd.DataFrame(X).copy()

        pre_model = RandomForestClassifier(
            n_estimators=50,
            random_state=self.random_state,
            n_jobs=-1
        )

        pre_model.fit(X, y)

        importances = pd.Series(
            pre_model.feature_importances_,
            index=X.columns
        )

        corr_matrix = X.corr(
            method="pearson"
        ).abs()

        columns = X.columns.tolist()

        to_drop = set()

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                col_a = columns[i]
                col_b = columns[j]

                if (
                        col_a in to_drop
                        or col_b in to_drop
                ):
                    continue

                corr_value = corr_matrix.loc[
                    col_a,
                    col_b
                ]

                if corr_value >= self.threshold:
                    weaker_feature = (
                        col_a
                        if importances[col_a]
                           < importances[col_b]
                        else col_b
                    )

                    to_drop.add(
                        weaker_feature
                    )

        self.to_drop_ = sorted(to_drop)

        self.selected_features_ = [
            c for c in columns
            if c not in self.to_drop_
        ]

        print(
            f"[*] CorrelationFilter removed "
            f"{len(self.to_drop_)} features."
        )

        return self

    # ======================================================
    # TRANSFORM
    # ======================================================

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        return X[self.selected_features_]
