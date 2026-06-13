import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

# Importy wizualizacji
from visualization.confusion_matrix_plot import plot_confusion_matrix
from visualization.feature_importance_plot import plot_feature_importance
from visualization.correlation_plot import plot_feature_correlation
from visualization.cv_scores_plot import plot_cv_scores


class RandomForestTrainer:

    def __init__(
            self,
            dataset_path,
            reports_dir="reports"
    ):
        self.dataset_path = dataset_path
        self.reports_dir = reports_dir

        self.model = None
        self.feature_names = None

        os.makedirs(
            self.reports_dir,
            exist_ok=True
        )

    # ======================================================
    # DATA LOADING & FILTERING
    # ======================================================

    def _filter_invalid_rows(self, df):
        required_cols = [
            "actual_packets_in_flow",
            "agg_value"
        ]

        if not all(col in df.columns for col in required_cols):
            print("[!] Missing columns for filtering.")
            return df

        before = len(df)
        df = df[df["actual_packets_in_flow"] == df["agg_value"]].copy()
        removed = before - len(df)

        print(f"[*] Removed {removed} invalid rows.")
        print(f"[+] Remaining rows: {len(df)}")

        return df

    def _load_data(self):
        print(f"[*] Loading data from {self.dataset_path}...")

        try:
            df = pd.read_csv(self.dataset_path)
            df = self._filter_invalid_rows(df)

            if df.empty:
                print("[!] Dataset is empty.")
                return None, None, None

            drop_cols = [
                c for c in ["label", "granularity", "agg_mode", "agg_value"]
                if c in df.columns
            ]

            X = df.drop(columns=drop_cols)
            y = df["label"]

            self.feature_names = list(X.columns)

            print(
                f"[+] Loaded {len(df)} samples | "
                f"{X.shape[1]} features | "
                f"{y.nunique()} classes"
            )

            return X, y, df

        except Exception as e:
            print(f"[!] Error loading data: {e}")
            return None, None, None

    # ======================================================
    # FEATURE SELECTION (CORRELATION)
    # ======================================================

    def _drop_correlated_features(self, X, y, threshold=0.85):
        """
        Delete features that are highly correlated, keeping the one with higher importance.
        MUST BE RUN ONLY ON TRAINING DATA to prevent data leakage.
        """
        print(f"[*] Running correlation filter (threshold={threshold}) on training data...")
        pre = RandomForestClassifier(n_estimators=50, random_state=32, n_jobs=-1)
        pre.fit(X, y)

        importances = pd.Series(pre.feature_importances_, index=X.columns)
        corr = X.corr(method="pearson").abs()
        cols = X.columns.tolist()
        to_drop = set()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if cols[i] in to_drop or cols[j] in to_drop:
                    continue
                if corr.loc[cols[i], cols[j]] >= threshold:
                    weaker = cols[i] if importances[cols[i]] < importances[cols[j]] else cols[j]
                    to_drop.add(weaker)

        dropped = sorted(to_drop)
        if dropped:
            print(f"[*] Deleted {len(dropped)} highly correlated features: {dropped}")

        return X.drop(columns=dropped), dropped

    # ======================================================
    # VALIDATION
    # ======================================================

    def validate_features(self):
        X, y, _ = self._load_data()

        if X is None:
            return

        print("\n" + "=" * 50)
        print(" FEATURE VALIDATION REPORT ")
        print("=" * 50)

        missing = X.isnull().sum()

        if missing.any():
            print(f"Features with missing values:\n{missing[missing > 0]}")
        else:
            print("[✓] No missing values.")

        variances = X.var()
        zero_var = variances[variances == 0].index.tolist()

        if zero_var:
            print(f"[!] Features with zero variance: {zero_var}")
        else:
            print("[✓] All features show variance.")

        return variances

    # ======================================================
    # TRAINING
    # ======================================================

    def train_and_evaluate(self):
        X, y, df = self._load_data()

        if X is None:
            return None

        gran_tag = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # --------------------------------------------------
        # TRAIN / TEST SPLIT (ZROBIONE PRZED SELEKCJĄ CECH!)
        # --------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=32,
            stratify=y
        )

        # --------------------------------------------------
        # FEATURE REDUCTION (TYLKO NA ZBIORZE TRENINGOWYM)
        # --------------------------------------------------
        X_train_unfiltered = X_train.copy()  # Zachowujemy kopię do wykresu korelacji

        X_train_filtered, dropped_features = self._drop_correlated_features(
            X_train,
            y_train,
            threshold=0.85
        )

        # Aplikujemy usunięcie kolumn na zbiorze testowym (BEZ sprawdzania korelacji)
        X_test_filtered = X_test.drop(columns=dropped_features)

        self.feature_names = list(X_train_filtered.columns)
        print(f"[*] Remaining features after reduction: {len(self.feature_names)}")

        # --------------------------------------------------
        # CLASS WEIGHTS
        # --------------------------------------------------
        classes = np.array(sorted(y_train.unique()))
        raw_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )
        class_weights = dict(zip(classes, raw_weights))

        # --------------------------------------------------
        # HYPERPARAMETER SEARCH
        # --------------------------------------------------
        rf = RandomForestClassifier(
            class_weight=class_weights,
            random_state=32,
            n_jobs=-1
        )

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 8, 12],
            "min_samples_split": [10, 20, 30],
            "min_samples_leaf": [5, 10, 15],
            "max_features": ["sqrt"]
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=32
        )

        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=15,
            scoring="f1_macro",
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=32
        )

        print("[*] Starting RandomizedSearchCV...")
        search.fit(X_train_filtered, y_train)

        self.model = search.best_estimator_

        print("\n[+] Best parameters:")
        print(search.best_params_)
        print(f"[+] Best CV score (from Search): {search.best_score_:.4f}")

        # --------------------------------------------------
        # PREDICTIONS ON TEST SET
        # --------------------------------------------------
        preds = self.model.predict(X_test_filtered)

        acc = accuracy_score(y_test, preds)

        print(f"\n[+] Result for {gran_tag}: Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, preds, digits=4))

        # --------------------------------------------------
        # CROSS VALIDATION (Na czystym zbiorze treningowym)
        # --------------------------------------------------
        cv_scores = cross_validate(
            self.model,
            X_train_filtered,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_weighted", "f1_macro"],
            n_jobs=-1
        )

        # --------------------------------------------------
        # VISUALIZATION
        # --------------------------------------------------
        plot_confusion_matrix(
            y_test,
            preds,
            sorted(y.unique()),
            self.reports_dir,
            gran_tag
        )

        plot_feature_importance(
            self.model,
            self.feature_names,
            self.reports_dir,
            gran_tag
        )

        # Do wykresu porównujemy X przed redukcją i po redukcji (ale z train setu!)
        plot_feature_correlation(
            X_train_unfiltered,
            X_train_filtered,
            self.reports_dir,
            gran_tag
        )

        plot_cv_scores(
            cv_scores,
            self.reports_dir,
            gran_tag
        )

        return acc

    # ======================================================
    # THRESHOLD PREDICTION
    # ======================================================

    def predict_with_threshold(self, X, threshold=0.60):
        if self.model is None:
            raise ValueError("Model is not loaded/trained.")

        # Upewniamy się, że zbiór X zawiera tylko poprawne cechy, jeśli X ma ich za dużo
        if isinstance(X, pd.DataFrame):
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in X: {missing_cols}")
            X_eval = X[self.feature_names]
        else:
            X_eval = X

        probs = self.model.predict_proba(X_eval)
        predictions = []

        for p in probs:
            max_prob = np.max(p)
            if max_prob < threshold:
                predictions.append("OTHER")
            else:
                predicted_class = self.model.classes_[np.argmax(p)]
                predictions.append(predicted_class)

        return predictions

    # ======================================================
    # SAVE MODEL
    # ======================================================

    def save_model(self, model_path):
        if self.model is None:
            print("[!] No model to save.")
            return

        payload = {
            "model": self.model,
            "feature_names": self.feature_names
        }

        joblib.dump(payload, model_path)
        print(f"[+] Model saved to: {model_path}")
