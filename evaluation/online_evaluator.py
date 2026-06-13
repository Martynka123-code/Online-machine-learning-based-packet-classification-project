# Pseudo-kod dla online_evaluator.py
class OnlineEvaluator:
    def __init__(self):
        self.current_label = None
        self.total_preds = 0
        self.correct_preds = 0

    def set_target_app(self, app_name):
        self.current_label = app_name
        print(f"[EVAL] Zmieniono cel na: {app_name}")

    def log_prediction(self, predicted_app):
        if self.current_label is None: return
        
        self.total_preds += 1
        if predicted_app.lower() == self.current_label.lower():
            self.correct_preds += 1
            
        acc = (self.correct_preds / self.total_preds) * 100
        print(f"[EVAL] Predykcja: {predicted_app} | True: {self.current_label} | Accuracy: {acc:.2f}%")