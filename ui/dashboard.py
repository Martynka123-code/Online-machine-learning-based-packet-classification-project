# Przykład struktury w nowym pliku: ui/dashboard.py
class OnlineDashboard:
    def __init__(self):
        self.target = "N/A"
        self.stats = {"correct": 0, "total": 0}

    def update_stats(self, prediction):
        self.stats["total"] += 1
        if prediction.lower() == self.target.lower():
            self.stats["correct"] += 1
        self.render()

    def set_target(self, new_target):
        self.target = new_target
        # Reset statystyk przy zmianie aplikacji
        self.stats = {"correct": 0, "total": 0} 
        self.render()

    def render(self):
        # Tutaj używasz print do rysowania "ładnego" UI
        acc = (self.stats["correct"] / self.stats["total"] * 100) if self.stats["total"] > 0 else 0
        print(f"\033[H\033[J") # Czyści ekran
        print(f"=== LIVE CLASSIFICATION DASHBOARD ===")
        print(f" Monitoring: {self.target.upper()}")
        print(f" Accuracy:   {acc:.2f}% ({self.stats['correct']}/{self.stats['total']})")
        print(f"=====================================")