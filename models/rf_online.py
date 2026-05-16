import joblib
import numpy as np
import pandas as pd
import warnings


class RandomForestOnline:

    def __init__(self, model_path, threshold=0.60):
        self.model_path = model_path
        self.feature_names = None
        self.model = None
        self.threshold = threshold

        try:
            payload = joblib.load(model_path)

            if isinstance(payload, dict) and "model" in payload:
                self.model = payload["model"]
                self.feature_names = payload.get("feature_names", None)
            else:
                self.model = payload

            print(f"[*] Loaded Random Forest model from: {model_path}")
            print(f"[*] Confidence threshold: {self.threshold}")

        except Exception as e:
            print(f"[!] Error loading RF model: {e}")

    # ----------------------------------------------------------

    def classify_stream(self, features_dict):
        """
        Predicts traffic class using confidence threshold.

        Returns:
            - predicted label
            - confidence
            - probability details
        """
        if self.model is None:
            return {
                "prediction": "No model loaded",
                "confidence": 0.0,
                "probabilities": {}
            }

        if not self.feature_names:
            return {
                "prediction": "Missing feature names",
                "confidence": 0.0,
                "probabilities": {}
            }

        # Budowanie wektora cech
        feature_vector = [features_dict.get(col, 0.0) for col in self.feature_names]
        X = [feature_vector]

        # Przewidywanie prawdopodobieństw
        probabilities = self.model.predict_proba(X)[0]
        max_prob = float(np.max(probabilities))
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = self.model.classes_[predicted_idx]

        # Logika progu ufności (Threshold)
        final_prediction = predicted_class if max_prob >= self.threshold else "OTHER"

        # Słownik prawdopodobieństw
        probability_dict = {
            cls: round(float(prob), 4)
            for cls, prob in zip(self.model.classes_, probabilities)
        }

        return {
            "prediction": final_prediction,
            "confidence": round(max_prob, 4),
            "probabilities": probability_dict
        }