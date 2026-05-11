# models/rf_trainer.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


class RandomForestTrainer:
    def __init__(self, dataset_path):
        """Initializes the trainer with the path to the aggregated feature CSV."""
        self.dataset_path = dataset_path
        # 100 trees and fixed seed for reproducibility
        self.model = RandomForestClassifier(n_estimators = 100, random_state = 42)

    def train_and_evaluate(self):
        """Loads data, trains the Random Forest model, and prints evaluation results.
        Returns the accuracy score (float).
        """
        print(f"[*] Loading data from {self.dataset_path}...")
        df = pd.DataFrame()
        try:
            df = pd.read_csv(self.dataset_path)
        except Exception as e:
            print(f"[!] Error during dataset loading: {e}")
            return None

        if df.empty:
            print("[!] Dataset is empty.")
            return None

        # Separate features (X) from target labels (y)
        X = df.drop(columns = ["label", "granularity"])
        y = df["label"]

        # Split into training (80%) and test (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        print("[*] Random Forest model training...")
        self.model.fit(X_train, y_train)

        print("[*] Evaluating model on the test set...")
        predictions = self.model.predict(X_test)

        # Feature Importance
        importances = pd.Series(self.model.feature_importances_, index = X.columns)
        print("\nMost important features:")
        print(importances.sort_values(ascending = False).head(5))

        acc = accuracy_score(y_test, predictions)
        print(f"\n=== Results (Accuracy: {acc * 100:.2f}%) ===")
        print(classification_report(y_test, predictions))

        return acc

    def save_model(self, model_path):
        """Serializes the trained model to disk for later use (in online_rf.py)."""
        joblib.dump(self.model, model_path)
        print(f"\n[+] Model saved to: {model_path}")