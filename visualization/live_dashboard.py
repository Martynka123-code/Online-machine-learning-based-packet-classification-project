"""
live_dashboard.py
------------------
Proste GUI (Tkinter) do monitorowania klasyfikacji online w czasie rzeczywistym.

Działa jednakowo dla RF i CNN — dashboard nie wie nic o modelu,
dostaje tylko (prediction, confidence, meta) przez kolejkę.

Użycie (w main.py):

    from visualization.live_dashboard import LiveAccuracyDashboard

    dashboard = LiveAccuracyDashboard(title="Live Monitor - RF")

    def on_flow_ready(features, flow_key):
        result = classifier.classify_stream(features)
        ...
        dashboard.push(result["prediction"], result["confidence"], meta="...")

    capture_thread = threading.Thread(target=sniffer.start_capture, daemon=True)
    capture_thread.start()

    dashboard.run()   # blokuje główny wątek, działa GUI
"""

import time
import queue
import tkinter as tk
from tkinter import ttk
from collections import Counter, deque


class LiveAccuracyDashboard:
    """
    GUI do śledzenia trafności klasyfikatora "na żywo".

    - Pokazuje ostatnio wykrytą aplikację i confidence.
    - Pozwala wpisać "aktualnie testowaną aplikację" (prawdziwa etykieta).
    - Po wpisaniu/zmianie etykiety statystyki sesji są resetowane i liczone od nowa.
    - Pokazuje accuracy, rozkład decyzji modelu i ostatnie N detekcji.
    """

    def __init__(self, title="Live Classification Monitor", history_len=15):
        self.queue = queue.Queue()
        self.history_len = history_len

        self.true_label = None
        self.session_total = 0
        self.session_correct = 0
        self.session_counts = Counter()
        self.history = deque(maxlen=history_len)

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("540x560")
        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------
    # UI BUILD
    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # --- Sekcja: ustawianie aktualnie testowanej aplikacji ---
        top = ttk.LabelFrame(self.root, text="Aktualnie testowana aplikacja")
        top.pack(fill="x", **pad)

        self.label_entry = ttk.Entry(top, font=("Segoe UI", 12))
        self.label_entry.pack(side="left", fill="x", expand=True, padx=8, pady=8)
        self.label_entry.bind("<Return>", lambda e: self._set_true_label())

        set_btn = ttk.Button(top, text="Ustaw / Reset stats", command=self._set_true_label)
        set_btn.pack(side="left", padx=8, pady=8)

        self.true_label_var = tk.StringVar(value="(nie ustawiono — accuracy nieliczone)")
        ttk.Label(self.root, textvariable=self.true_label_var,
                  font=("Segoe UI", 10, "italic")).pack(anchor="w", padx=12)

        # --- Sekcja: ostatnia detekcja ---
        last = ttk.LabelFrame(self.root, text="Ostatnia detekcja")
        last.pack(fill="x", **pad)

        self.last_pred_var = tk.StringVar(value="-")
        self.last_conf_var = tk.StringVar(value="-")
        self.last_meta_var = tk.StringVar(value="-")

        ttk.Label(last, text="Wykryta aplikacja:").grid(row=0, column=0, sticky="w", padx=8, pady=2)
        ttk.Label(last, textvariable=self.last_pred_var,
                  font=("Segoe UI", 16, "bold")).grid(row=0, column=1, sticky="w", padx=8, pady=2)

        ttk.Label(last, text="Confidence:").grid(row=1, column=0, sticky="w", padx=8, pady=2)
        ttk.Label(last, textvariable=self.last_conf_var,
                  font=("Segoe UI", 12)).grid(row=1, column=1, sticky="w", padx=8, pady=2)

        ttk.Label(last, text="Szczegóły:").grid(row=2, column=0, sticky="w", padx=8, pady=2)
        ttk.Label(last, textvariable=self.last_meta_var,
                  font=("Segoe UI", 9)).grid(row=2, column=1, sticky="w", padx=8, pady=2)

        # --- Sekcja: statystyki sesji ---
        stats = ttk.LabelFrame(self.root, text="Statystyki bieżącej sesji")
        stats.pack(fill="x", **pad)

        self.total_var = tk.StringVar(value="0")
        self.correct_var = tk.StringVar(value="0")
        self.acc_var = tk.StringVar(value="-")

        ttk.Label(stats, text="Łącznie próbek:").grid(row=0, column=0, sticky="w", padx=8, pady=2)
        ttk.Label(stats, textvariable=self.total_var).grid(row=0, column=1, sticky="w", padx=8, pady=2)

        ttk.Label(stats, text="Poprawne (== target):").grid(row=1, column=0, sticky="w", padx=8, pady=2)
        ttk.Label(stats, textvariable=self.correct_var).grid(row=1, column=1, sticky="w", padx=8, pady=2)

        ttk.Label(stats, text="Accuracy:").grid(row=2, column=0, sticky="w", padx=8, pady=2)
        ttk.Label(stats, textvariable=self.acc_var,
                  font=("Segoe UI", 12, "bold")).grid(row=2, column=1, sticky="w", padx=8, pady=2)

        # --- Sekcja: rozkład predykcji ---
        dist = ttk.LabelFrame(self.root, text="Rozkład decyzji modelu (sesja)")
        dist.pack(fill="both", expand=True, **pad)

        self.dist_tree = ttk.Treeview(
            dist, columns=("class", "count", "pct"), show="headings", height=6
        )
        self.dist_tree.heading("class", text="Klasa")
        self.dist_tree.heading("count", text="Liczba")
        self.dist_tree.heading("pct", text="%")
        self.dist_tree.column("class", width=180, anchor="w")
        self.dist_tree.column("count", width=80, anchor="center")
        self.dist_tree.column("pct", width=80, anchor="center")
        self.dist_tree.pack(fill="both", expand=True, padx=8, pady=4)

        # --- Sekcja: historia ---
        hist = ttk.LabelFrame(self.root, text=f"Ostatnie {self.history_len} detekcji")
        hist.pack(fill="both", expand=True, **pad)

        self.hist_text = tk.Text(hist, height=8, state="disabled", font=("Consolas", 9))
        self.hist_text.pack(fill="both", expand=True, padx=8, pady=4)

    # ------------------------------------------------------------------
    # OBSŁUGA ZMIANY ETYKIETY
    # ------------------------------------------------------------------
    def _set_true_label(self):
        new_label = self.label_entry.get().strip().lower()
        if not new_label:
            return

        self.true_label = new_label
        self.true_label_var.set(f"Testowana aplikacja: {new_label.upper()}  (statystyki zresetowane)")

        # Reset statystyk sesji — nowa aplikacja = nowy pomiar od zera
        self.session_total = 0
        self.session_correct = 0
        self.session_counts.clear()
        self.history.clear()

        self._refresh_stats()
        self._refresh_history()

    # ------------------------------------------------------------------
    # API DLA WĄTKU SNIFFERA
    # ------------------------------------------------------------------
    def push(self, prediction, confidence, meta=None):
        """Wywoływane z wątku sniffera (thread-safe — przez kolejkę)."""
        self.queue.put((prediction, confidence, meta))

    # ------------------------------------------------------------------
    # PĘTLA ODCZYTU KOLEJKI (działa na wątku GUI)
    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                prediction, confidence, meta = self.queue.get_nowait()
                self._handle_prediction(prediction, confidence, meta)
        except queue.Empty:
            pass

        self.root.after(150, self._poll_queue)

    # ------------------------------------------------------------------
    def _handle_prediction(self, prediction, confidence, meta):
        pred_norm = str(prediction).strip().lower()

        self.last_pred_var.set(str(prediction).upper())
        self.last_conf_var.set(f"{confidence * 100:.1f}%")
        self.last_meta_var.set(str(meta) if meta else "-")

        self.session_total += 1
        self.session_counts[pred_norm] += 1

        if self.true_label is not None and pred_norm == self.true_label:
            self.session_correct += 1

        ts = time.strftime("%H:%M:%S")
        extra = f" | {meta}" if meta else ""
        self.history.append(f"[{ts}] {prediction} ({confidence * 100:.1f}%){extra}")

        self._refresh_stats()
        self._refresh_history()

    # ------------------------------------------------------------------
    def _refresh_stats(self):
        self.total_var.set(str(self.session_total))
        self.correct_var.set(str(self.session_correct))

        if self.true_label is not None and self.session_total > 0:
            acc = self.session_correct / self.session_total * 100
            self.acc_var.set(f"{acc:.2f}%")
        else:
            self.acc_var.set("-")

        for row in self.dist_tree.get_children():
            self.dist_tree.delete(row)

        for cls, cnt in self.session_counts.most_common():
            pct = cnt / self.session_total * 100 if self.session_total else 0
            label = cls.upper()
            if cls == self.true_label:
                label += "  <-- target"
            self.dist_tree.insert("", "end", values=(label, cnt, f"{pct:.1f}%"))

    # ------------------------------------------------------------------
    def _refresh_history(self):
        self.hist_text.configure(state="normal")
        self.hist_text.delete("1.0", "end")
        for line in reversed(self.history):
            self.hist_text.insert("end", line + "\n")
        self.hist_text.configure(state="disabled")

    # ------------------------------------------------------------------
    def run(self):
        """Blokuje wątek wywołujący do zamknięcia okna GUI."""
        self.root.mainloop()