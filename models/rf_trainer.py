# models/rf_trainer.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from visualization.confusion_matrix_plot import plot_confusion_matrix
from visualization.feature_importance_plot import plot_feature_importance
from visualization.correlation_plot import plot_feature_correlation
from visualization.cv_scores_plot import plot_cv_scores

class RandomForestTrainer:
    def __init__(self, dataset_path, reports_dir="reports"):
        self.dataset_path = dataset_path
        self.reports_dir = reports_dir
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None

        os.makedirs(self.reports_dir, exist_ok=True)

    def _load_data(self):
        """Loading data with error handling."""
        print(f"[*] Loading data from {self.dataset_path}...")
        try:
            df = pd.read_csv(self.dataset_path)
            if df.empty:
                print("[!] Dataset is empty.")
                return None, None, None
            
            drop_cols = [c for c in ["label", "granularity"] if c in df.columns]
            X = df.drop(columns=drop_cols)
            y = df["label"]
            self.feature_names = list(X.columns)

            print(f"[+] Loaded {len(df)} samples | {X.shape[1]} features | {y.nunique()} classes")
            return X, y, df
        except Exception as e:
            print(f"[!] Error loading data: {e}")
            return None, None, None

    def _drop_correlated_features(self, X, y, threshold=0.85):
        """Delete features that are highly correlated, keeping the one with higher importance."""
        pre = RandomForestClassifier(n_estimators=50, random_state=42)
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

    def validate_features(self):
        """Generates a report on feature quality, including missing values and zero variance."""
        X, y, _ = self._load_data()
        if X is None: return

        print("\n" + "="*50 + "\n FEATURE VALIDATION REPORT \n" + "="*50)
        missing = X.isnull().sum()
        if missing.any():
            print(f"Features with missing values:\n{missing[missing > 0]}")
        else:
            print("None of the features have missing values.")

        variances = X.var()
        zero_var = variances[variances == 0].index.tolist()
        if zero_var:
            print(f"[!] Features with zero variance (to be removed): {zero_var}")
        else:
            print("[✓] All features show variance.")
        return variances

    def train_and_evaluate(self):
        """Fit model, evaluate on test set, and generate reports."""
        X, y, df = self._load_data()
        if X is None: return None

        classes = np.array(sorted(y.unique()))
        raw_weights = compute_class_weight("balanced", classes=classes, y=y)
        cw_dict = dict(zip(classes, raw_weights))

        gran_tag = os.path.splitext(os.path.basename(self.dataset_path))[0]

        X_before = X.copy()
        X, _ = self._drop_correlated_features(X, y)
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        cv_model = RandomForestClassifier(n_estimators=100, class_weight=cw_dict, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(cv_model, X, y, cv=cv, scoring=["accuracy", "f1_weighted", "f1_macro"])
        self.model = RandomForestClassifier(n_estimators=100, class_weight=cw_dict, random_state=42)
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"\n[+] Result for {gran_tag}: Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, preds, digits=4))

        plot_confusion_matrix(y_test, preds, sorted(y.unique()), self.reports_dir, gran_tag)
        plot_feature_importance(self.model, self.feature_names, self.reports_dir, gran_tag)
        plot_feature_correlation(X_before, X_train, self.reports_dir, gran_tag)
        plot_cv_scores(cv_res, self.reports_dir, gran_tag)

        return acc

    def save_model(self, model_path):
        """Save the trained model along with the label encoder and feature names."""
        if self.model is None:
            print("[!] There is no model to save.")
            return

        payload = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, model_path)
        print(f"[+] Model saved to: {model_path}")