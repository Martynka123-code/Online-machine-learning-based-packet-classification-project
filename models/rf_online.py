import joblib
import pandas as pd
import warnings

class RandomForestOnline:
    def __init__(self, model_path):
        self.model_path = model_path
        try:
            self.model = joblib.load(model_path)
            print(f"[*] Loaded Random Forest model from: {model_path}")
        except Exception as e:
            print(f"[!] Error loading RF model: {e}")
            self.model = None

    def classify_stream(self, features_dict):
        """
        Takes a dictionary of features extracted by FlowFeatureExtractor
        and returns the predicted class (e.g., 'Spotify', 'YouTube').
        """
        if self.model is None:
            return "No model loaded"
            
        # Ignore warnings (scikit-learn sometimes complains about missing column names for single predictions)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Convert dictionary to DataFrame, as required by Random Forest
            df = pd.DataFrame([features_dict])
            
            # Drop labels if they somehow ended up here (the model only accepts features)
            if 'label' in df.columns:
                df = df.drop(columns=['label'])
            if 'granularity' in df.columns:
                df = df.drop(columns=['granularity'])

            prediction = self.model.predict(df)
            return prediction[0]