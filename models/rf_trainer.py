# models/rf_trainer.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

class RandomForestTrainer:
    def __init__(self, dataset_path, reports_dir="reports"):
        self.dataset_path = dataset_path
        self.reports_dir = reports_dir
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.feature_names = None

    def train_and_evaluate(self):
        """Loads data, performs training and returns accuracy."""
        try:
            df = pd.read_csv(self.dataset_path)
            if df.empty: return None
            
            X = df.drop(columns=["label", "granularity"])
            y = df["label"]
            self.feature_names = X.columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            
            predictions = self.model.predict(X_test)
            return accuracy_score(y_test, predictions)
        except Exception as e:
            print(f"[!] Training error: {e}")
            return None

    def save_model(self, model_path):
        """Saves model with metadata (Agata's style)."""
        payload = {
            "model": self.model,
            "feature_names": self.feature_names
        }
        joblib.dump(payload, model_path)
        print(f"[*] Model saved to: {model_path}")