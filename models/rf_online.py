import joblib
import pandas as pd
import warnings

class RandomForestOnline:
    def __init__(self, model_path):
        self.model_path = model_path
        self.feature_names = None
        self.model = None
        
        try:
            payload = joblib.load(model_path)
            
            if isinstance(payload, dict) and "model" in payload:
                self.model = payload["model"]
                self.feature_names = payload.get("feature_names", None)
            else:
                self.model = payload 
                
            print(f"[*] Loaded Random Forest model from: {model_path}")
        except Exception as e:
            print(f"[!] Error loading RF model: {e}")

    def classify_stream(self, features_dict):
        """
        Takes a dictionary of features extracted by FlowFeatureExtractor
        and returns the predicted class (e.g., 'Spotify', 'YouTube').
        """
        if self.model is None or not self.feature_names:
            return "No model loaded"
            
        feature_vector = []
        for col in self.feature_names:
            feature_vector.append(features_dict.get(col, 0.0))

        prediction = self.model.predict([feature_vector])
        return prediction[0]