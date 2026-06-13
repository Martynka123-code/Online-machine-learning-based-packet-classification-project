"""
report_generator.py
-------------------
Formatowanie i wyświetlanie raportów ewaluacji w konsoli.
Obsługuje zarówno pojedynczy plik, jak i raport zbiorczy z wielu PCAP.
"""

import numpy as np
from collections import Counter, defaultdict

W = 60  # szerokość raportu


def _bar(value, total, width=30):
    """Prosta pasek procentowy ASCII."""
    filled = int(round(value / total * width)) if total > 0 else 0
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def print_single_report(stats):
    """
    Drukuje raport dla jednego pliku PCAP.

    Args:
        stats (dict): Słownik zwrócony przez evaluate_pcap()
    """
    if stats is None:
        print("  [!] Brak danych do raportu.")
        return

    total = stats["total"]
    correct = stats["correct"]
    wrong = stats["wrong"]
    other = stats["other"]
    counts = stats["counts"]

    print()
    print("=" * W)
    print(" RAPORT EWALUACJI ".center(W))
    print("=" * W)
    print(f"  Plik:             {stats['pcap_name']}")
    print(f"  Prawdziwa klasa:  {stats['true_label'].upper()}")
    print(f"  Model:            {stats['model_type'].upper()}", end="")
    if stats["model_type"] == "rf":
        p = stats["model_params"]
        print(f"  [{p['agg_mode']}, val={p['agg_value']}]")
    else:
        print()
    print(f"  Łączna liczba próbek: {total}")
    print("-" * W)

    print(f"\n  {'POPRAWNIE:':<14} {correct:>5}  {_bar(correct, total)}  {correct / total * 100:>6.2f}%")
    print(f"  {'BŁĘDNIE:':<14} {wrong:>5}  {_bar(wrong, total)}  {wrong / total * 100:>6.2f}%")
    if other > 0:
        print(f"  {'OTHER/UNKNOWN:':<14} {other:>5}  {_bar(other, total)}  {other / total * 100:>6.2f}%")

    print()
    print("  Rozkład decyzji modelu:")
    print("  " + "-" * (W - 2))
    for cls, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- prawdziwa" if cls == stats["true_label"] else ""
        print(f"  {cls.upper():<18} {cnt:>5}  {_bar(cnt, total)}  {cnt / total * 100:>6.2f}%{marker}")

    # Confidence (tylko RF)
    if stats["model_type"] == "rf" and stats["class_confidence"]:
        print()
        print("  Analiza pewności (confidence) - tylko RF:")
        print("  " + "-" * (W - 2))
        print(f"  {'Średnia ogólna:':<22} {stats['mean_confidence'] * 100:>6.1f}%")
        for cls, conf in sorted(stats["class_confidence"].items(),
                                key=lambda x: x[1], reverse=True):
            print(f"  > {cls.upper():<20} {conf * 100:>6.1f}%")

    # Confidence (CNN — mean per plik)
    if stats["model_type"] == "cnn" and stats.get("mean_confidence", 0) > 0:
        print()
        print("  Analiza pewności (confidence) - CNN:")
        print("  " + "-" * (W - 2))
        print(f"  {'Średnia ogólna:':<22} {stats['mean_confidence'] * 100:>6.1f}%")

    print("=" * W)
    print()


def print_batch_report(all_stats):
    """
    Drukuje zbiorczy raport po ewaluacji wielu plików PCAP.

    Args:
        all_stats (list[dict]): Lista słowników z evaluate_pcap()
    """
    valid = [s for s in all_stats if s is not None]
    if not valid:
        print("[!] Brak wyników do raportu zbiorczego.")
        return

    total_samples = sum(s["total"] for s in valid)
    total_correct = sum(s["correct"] for s in valid)
    total_wrong = sum(s["wrong"] for s in valid)
    total_other = sum(s["other"] for s in valid)

    label_stats = defaultdict(lambda: {"total": 0, "correct": 0, "counts": Counter()})
    for s in valid:
        lb = s["true_label"]
        label_stats[lb]["total"] += s["total"]
        label_stats[lb]["correct"] += s["correct"]
        for cls, cnt in s["counts"].items():
            label_stats[lb]["counts"][cls] += cnt

    print()
    print("=" * W)
    print(" RAPORT ZBIORCZY ".center(W))
    print("=" * W)
    print(f"  Liczba plików PCAP: {len(valid)}")
    print(f"  Łączna liczba próbek: {total_samples}")
    print("-" * W)
    print(f"\n  {'POPRAWNIE:':<14} {total_correct:>6}  {_bar(total_correct, total_samples)}  {total_correct / total_samples * 100:>6.2f}%")
    print(f"  {'BŁĘDNIE:':<14} {total_wrong:>6}  {_bar(total_wrong, total_samples)}  {total_wrong / total_samples * 100:>6.2f}%")
    if total_other > 0:
        print(f"  {'OTHER/UNKNOWN:':<14} {total_other:>6}  {_bar(total_other, total_samples)}  {total_other / total_samples * 100:>6.2f}%")

    print()
    print("  Dokładność per-klasa:")
    print("  " + "-" * (W - 2))
    for label, data in sorted(label_stats.items()):
        t = data["total"]
        c = data["correct"]
        acc = c / t * 100 if t > 0 else 0.0
        print(f"  {label.upper():<18} {c:>5}/{t:<5}  {_bar(c, t)}  {acc:>6.2f}%")

    print()
    print("  Rozkład decyzji per-klasa (gdzie trafiały pakiety):")
    print("  " + "-" * (W - 2))
    for label, data in sorted(label_stats.items()):
        t = data["total"]
        print(f"\n  Prawdziwa klasa: {label.upper()} ({t} próbek)")
        for cls, cnt in sorted(data["counts"].items(), key=lambda x: x[1], reverse=True):
            marker = " <--" if cls == label else ""
            print(f"    {cls.upper():<18} {cnt:>5}  {_bar(cnt, t, 20)}  {cnt / t * 100:>6.2f}%{marker}")

    # Confidence zbiorcze (RF)
    rf_stats = [s for s in valid if s["model_type"] == "rf" and s["class_confidence"]]
    if rf_stats:
        agg_conf = defaultdict(list)
        for s in rf_stats:
            for cls, conf in s["class_confidence"].items():
                agg_conf[cls].append(conf)
        print()
        print("  Średnia pewność (confidence) po klasach - RF:")
        print("  " + "-" * (W - 2))
        for cls, confs in sorted(agg_conf.items(), key=lambda x: np.mean(x[1]), reverse=True):
            print(f"  > {cls.upper():<20} {np.mean(confs) * 100:>6.1f}%")

    # Confidence zbiorcze (CNN)
    cnn_stats = [s for s in valid if s["model_type"] == "cnn" and s.get("mean_confidence", 0) > 0]
    if cnn_stats:
        mean_cnn_conf = np.mean([s["mean_confidence"] for s in cnn_stats])
        print()
        print("  Średnia pewność (confidence) - CNN:")
        print("  " + "-" * (W - 2))
        print(f"  Średnia ogólna: {mean_cnn_conf * 100:>6.1f}%")

    print()
    print("  Wyniki per-plik:")
    print("  " + "-" * (W - 2))
    for s in valid:
        t = s["total"]
        c = s["correct"]
        print(f"  {s['pcap_name']:<30} {c:>4}/{t:<4}  {c / t * 100:>6.2f}%  [{s['true_label'].upper()}]")

    print("=" * W)
    print()


def print_comparison_report(rf_stats_list, cnn_stats_list):
    """
    Porównuje wyniki RF i CNN na tych samych plikach PCAP.

    Args:
        rf_stats_list (list[dict]): Wyniki z evaluate_pcap() dla RF
        cnn_stats_list (list[dict]): Wyniki z evaluate_pcap() dla CNN
    """
    rf_valid = [s for s in rf_stats_list if s is not None]
    cnn_valid = [s for s in cnn_stats_list if s is not None]

    if not rf_valid and not cnn_valid:
        print("[!] Brak danych do porównania.")
        return

    print()
    print("=" * W)
    print(" PORÓWNANIE RF vs CNN ".center(W))
    print("=" * W)

    # Zbiorcze accuracy
    def _acc(stats_list):
        total = sum(s["total"] for s in stats_list)
        correct = sum(s["correct"] for s in stats_list)
        return correct / total * 100 if total > 0 else 0.0

    rf_acc = _acc(rf_valid) if rf_valid else None
    cnn_acc = _acc(cnn_valid) if cnn_valid else None

    print(f"\n  {'Model':<12} {'Accuracy':>10}  {'Próbki':>8}")
    print("  " + "-" * (W - 2))
    if rf_acc is not None:
        rf_total = sum(s["total"] for s in rf_valid)
        print(f"  {'RF':<12} {rf_acc:>9.2f}%  {rf_total:>8}")
    if cnn_acc is not None:
        cnn_total = sum(s["total"] for s in cnn_valid)
        print(f"  {'CNN':<12} {cnn_acc:>9.2f}%  {cnn_total:>8}")

    if rf_acc is not None and cnn_acc is not None:
        diff = cnn_acc - rf_acc
        winner = "CNN" if diff > 0 else "RF"
        print(f"\n  Różnica: {abs(diff):.2f}% na korzyść {winner}")

    # Per-klasa porównanie
    def _per_class(stats_list):
        per = defaultdict(lambda: {"total": 0, "correct": 0})
        for s in stats_list:
            lb = s["true_label"]
            per[lb]["total"] += s["total"]
            per[lb]["correct"] += s["correct"]
        return per

    rf_per = _per_class(rf_valid) if rf_valid else {}
    cnn_per = _per_class(cnn_valid) if cnn_valid else {}
    all_labels = sorted(set(list(rf_per.keys()) + list(cnn_per.keys())))

    if all_labels:
        print()
        print("  Dokładność per-klasa:")
        print("  " + "-" * (W - 2))
        print(f"  {'Klasa':<18} {'RF':>8}  {'CNN':>8}  {'Lepszy':>8}")
        print("  " + "-" * (W - 2))
        for label in all_labels:
            rf_d = rf_per.get(label, {})
            cnn_d = cnn_per.get(label, {})
            rf_a = rf_d["correct"] / rf_d["total"] * 100 if rf_d.get("total", 0) > 0 else None
            cnn_a = cnn_d["correct"] / cnn_d["total"] * 100 if cnn_d.get("total", 0) > 0 else None

            rf_str = f"{rf_a:>6.2f}%" if rf_a is not None else "     N/A"
            cnn_str = f"{cnn_a:>6.2f}%" if cnn_a is not None else "     N/A"

            if rf_a is not None and cnn_a is not None:
                better = "CNN ✓" if cnn_a > rf_a else ("RF ✓" if rf_a > cnn_a else "  remis")
            else:
                better = ""

            print(f"  {label.upper():<18} {rf_str}  {cnn_str}  {better}")

    print("=" * W)
    print()