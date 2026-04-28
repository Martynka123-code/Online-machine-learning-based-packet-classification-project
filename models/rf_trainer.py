# models/rf_trainer.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


class RandomForestTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_and_evaluate(self):
        print(f"[*] Wczytywanie danych z {self.dataset_path}...")
        df = pd.DataFrame()
        try:
            df = pd.read_csv(self.dataset_path)
        except Exception as e:
            print(f"[!] Błąd wczytywania danych: {e}")
            return

        if df.empty:
            print("[!] Zbiór danych jest pusty.")
            return

        # Rozdzielenie cech (X) od etykiet (y)
        X = df.drop(columns=["label", "granularity"])
        y = df["label"]

        # Podział na zbiór treningowy i testowy (80% / 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("[*] Trenowanie modelu Random Forest...")
        self.model.fit(X_train, y_train)

        print("[*] Ocena modelu na zbiorze testowym...")
        predictions = self.model.predict(X_test)

        # Wypisanie najważniejszych cech (Feature Importance)
        importances = pd.Series(self.model.feature_importances_, index=X.columns)
        print("\nNajważniejsze cechy:")
        print(importances.sort_values(ascending=False).head(5))

        # (reszta Twojego kodu z rf_trainer.py...)
        acc = accuracy_score(y_test, predictions)
        print(f"\n=== WYNIKI (Accuracy: {acc * 100:.2f}%) ===")
        print(classification_report(y_test, predictions))

        importances = pd.Series(self.model.feature_importances_, index=X.columns)
        print("\nNajważniejsze cechy:")
        print(importances.sort_values(ascending=False).head(5))

        # DODAJ TĘ JEDNĄ LINIJKĘ NA KOŃCU:
        return acc

    def save_model(self, model_path):
        """Zapisuje wytrenowany model do pliku, by użyć go później w online_rf.py"""
        joblib.dump(self.model, model_path)
        print(f"\n[+] Model zapisany w: {model_path}")