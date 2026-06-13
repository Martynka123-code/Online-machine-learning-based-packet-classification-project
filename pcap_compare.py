"""
pcap_compare.py
===============
Standalone PCAP comparison analysis tool for Incoming (BWD) Traffic.

Usage:
    python pcap_compare.py                  # reads from data/pcap_to_compare/
    python pcap_compare.py --input my/folder   # custom input folder
    python pcap_compare.py --gran 50               # custom packet granularity
    python pcap_compare.py --output reports/cmp    # custom output folder
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from scapy.utils import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Colour palette (one colour per class)
# ─────────────────────────────────────────────
PALETTE = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#3F51B5", "#F44336",
    "#009688", "#FFC107",
]


# ═══════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════

def get_flow_key(packet):
    """Direction-agnostic 5-tuple."""
    if packet.haslayer(IP):
        ip = packet[IP]; proto = ip.proto
    elif packet.haslayer(IPv6):
        ip = packet[IPv6]; proto = ip.nh
    else:
        return None

    if packet.haslayer(TCP):
        sp, dp = packet[TCP].sport, packet[TCP].dport
    elif packet.haslayer(UDP):
        sp, dp = packet[UDP].sport, packet[UDP].dport
    else:
        return None

    ips   = tuple(sorted([ip.src, ip.dst]))
    ports = tuple(sorted([sp, dp]))
    return (ips[0], ips[1], ports[0], ports[1], proto)


def get_payload_len(packet):
    if packet.haslayer(TCP) and packet[TCP].payload:
        return len(packet[TCP].payload)
    if packet.haslayer(UDP) and packet[UDP].payload:
        return len(packet[UDP].payload)
    return 0


TCP_FLAG_MASKS = {"SYN": 0x02, "ACK": 0x10, "FIN": 0x01,
                  "RST": 0x04, "PSH": 0x08, "URG": 0x20}


def direction_stats(pkts, prefix):
    """Compute per-direction statistical features."""
    if not pkts:
        keys = ["pkt_count", "pkt_len_mean", "pkt_len_std", "pkt_len_p50",
                "pkt_len_p75", "pkt_len_max", "payload_median",
                "iat_mean", "iat_std", "iat_median", "iat_cov"]
        return {f"{prefix}_{k}": 0.0 for k in keys}

    lens     = [len(p) for p in pkts]
    payloads = [get_payload_len(p) for p in pkts]
    iats     = [float(pkts[i].time - pkts[i-1].time) for i in range(1, len(pkts))]
    if not iats:
        iats = [0.0]

    iat_mean = float(np.mean(iats))
    iat_std  = float(np.std(iats))

    return {
        f"{prefix}_pkt_count":      len(pkts),
        f"{prefix}_pkt_len_mean":   float(np.mean(lens)),
        f"{prefix}_pkt_len_std":    float(np.std(lens)),
        f"{prefix}_pkt_len_p50":    float(np.percentile(lens, 50)),
        f"{prefix}_pkt_len_p75":    float(np.percentile(lens, 75)),
        f"{prefix}_pkt_len_max":    float(np.max(lens)),
        f"{prefix}_payload_median": float(np.median(payloads)),
        f"{prefix}_iat_mean":       iat_mean,
        f"{prefix}_iat_std":        iat_std,
        f"{prefix}_iat_median":     float(np.median(iats)),
        f"{prefix}_iat_cov":        (iat_std / iat_mean) if iat_mean > 0 else 0.0,
    }


def calculate_features(packets, label, local_ip):
    """Full feature vector matching RandomForestTrainer's pipeline (FIXED DIRECTION STABILITY)."""
    if not packets:
        return None

    fwd, bwd = [], []
    for p in packets:
        if p.haslayer(IP):
            src_ip = p[IP].src
        elif p.haslayer(IPv6):
            src_ip = p[IPv6].src
        else:
            continue
        
        # POPRAWKA: Jeśli pakiet pochodzi z lokalnego IP komputera -> zawsze FWD. W przeciwnym razie BWD.
        if src_ip == local_ip:
            fwd.append(p)
        else:
            bwd.append(p)

    feats = {}
    feats.update(direction_stats(fwd, "fwd"))
    feats.update(direction_stats(bwd, "bwd"))

    fwd_b = sum(len(p) for p in fwd)
    bwd_b = sum(len(p) for p in bwd)
    tot_b = fwd_b + bwd_b
    feats["bytes_ratio_fwd"] = (fwd_b / tot_b) if tot_b > 0 else 0.5
    feats["pkt_ratio_fwd"]   = (len(fwd) / len(packets)) if packets else 0.5

    n = max(len(packets), 1)
    flag_counts = {f: 0 for f in TCP_FLAG_MASKS}
    seen_seqs, retrans = set(), 0
    udp_count = 0
    for p in packets:
        if p.haslayer(TCP):
            fl = p[TCP].flags
            for fname, mask in TCP_FLAG_MASKS.items():
                if fl & mask:
                    flag_counts[fname] += 1
            seq = p[TCP].seq
            if seq in seen_seqs:
                retrans += 1
            else:
                seen_seqs.add(seq)
        elif p.haslayer(UDP):
            udp_count += 1

    for fname, cnt in flag_counts.items():
        feats[f"tcp_{fname.lower()}_ratio"] = cnt / n
    feats["tcp_retrans_ratio"] = retrans / n
    feats["udp_ratio"]         = udp_count / n

    # Burst detection (100 ms threshold)
    times = sorted(float(p.time) for p in packets)
    if len(times) > 1:
        gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
        lg   = [g for g in gaps if g > 0.1]
        feats["burst_count"]    = len(lg) + 1
        feats["mean_burst_gap"] = float(np.mean(lg)) if lg else 0.0
    else:
        feats["burst_count"]    = 1
        feats["mean_burst_gap"] = 0.0

    feats["actual_packets_in_flow"] = len(packets)
    feats["label"] = label
    
    if packets:
        feats["flow_id"] = str(get_flow_key(packets[0]))
    
    return feats


def pcap_to_dataframe(pcap_path: str, label: str, gran: int) -> pd.DataFrame:
    """Read a PCAP file and return a DataFrame of flow aggregates with Auto-IP detection."""
    flows   = {}
    records = []

    print(f"  ↳ {Path(pcap_path).name}  (class='{label}', gran={gran})")

    # KROK 1: Szybki pierwszy przebieg, aby znaleźć najczęstszy IP (IP Twojego komputera)
    ip_counts = {}
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if pkt.haslayer(IP):
                ip_counts[pkt[IP].src] = ip_counts.get(pkt[IP].src, 0) + 1
                ip_counts[pkt[IP].dst] = ip_counts.get(pkt[IP].dst, 0) + 1
            elif pkt.haslayer(IPv6):
                ip_counts[pkt[IPv6].src] = ip_counts.get(pkt[IPv6].src, 0) + 1
                ip_counts[pkt[IPv6].dst] = ip_counts.get(pkt[IPv6].dst, 0) + 1

    if not ip_counts:
        return pd.DataFrame()
        
    # Adres IP o największej liczbie wystąpień to nasz localhost
    local_ip = max(ip_counts, key=ip_counts.get)
    print(f"      [info] Auto-detected local client IP: {local_ip}")

    # KROK 2: Właściwa ekstrakcja cech z użyciem wykrytego stałego lokalnego IP
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            key = get_flow_key(pkt)
            if key is None:
                continue
            if key not in flows:
                flows[key] = []
            flows[key].append(pkt)

            if len(flows[key]) >= gran:
                row = calculate_features(flows[key], label, local_ip)
                if row:
                    records.append(row)
                flows[key] = []

    # flush partial flows (>= 5 packets)
    for pkts in flows.values():
        if len(pkts) >= 5:
            row = calculate_features(pkts, label, local_ip)
            if row:
                records.append(row)

    df = pd.DataFrame(records)
    print(f"      → {len(df)} flow aggregates extracted")
    return df


# ═══════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════

def compute_summary(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    rows = []
    for cls in sorted(df["label"].unique()):
        sub = df[df["label"] == cls][feat_cols]
        for col in feat_cols:
            rows.append({
                "class":    cls,
                "feature":  col,
                "mean":     sub[col].mean(),
                "std":      sub[col].std(),
                "median":   sub[col].median(),
                "p25":      sub[col].quantile(0.25),
                "p75":      sub[col].quantile(0.75),
                "n":        len(sub),
            })
    return pd.DataFrame(rows)


def compute_discrimination(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    rows = []
    classes = df["label"].unique()
    n_total = len(df)

    for col in feat_cols:
        groups = [df[df["label"] == c][col].dropna().values for c in classes]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        try:
            stat, p = stats.kruskal(*groups)
            eta2 = (stat - len(groups) + 1) / (n_total - len(groups))
            eta2 = max(0.0, eta2)
        except Exception:
            stat, p, eta2 = np.nan, np.nan, 0.0

        rows.append({"feature": col, "H_stat": stat, "p_value": p, "eta2": eta2})

    if not rows:
        print("\n[!] UWAGA: Zbyt mało klas lub cech do obliczenia testu Kruskala-Wallis.")
        return pd.DataFrame(columns=["feature", "H_stat", "p_value", "eta2"])

    return pd.DataFrame(rows).sort_values("eta2", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════

def class_colours(labels):
    unique = sorted(set(labels))
    return {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(unique)}


def plot_class_distribution(df, out_dir):
    counts = df["label"].value_counts().sort_index()
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(max(6, len(counts) * 1.2), 4))
    bars = ax.bar(counts.index, counts.values, color=colours, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Application class", fontsize=11)
    ax.set_ylabel("Flow aggregate count", fontsize=11)
    ax.set_title("Sample Distribution per Class (Balanced)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_discrimination_bar(disc_df, out_dir, top_n=20):
    top = disc_df.head(top_n)
    if top.empty: return

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.45)))
    colours = ["#2196F3" if i < 5 else "#90CAF9" for i in range(len(top))]
    ax.barh(top["feature"][::-1], top["eta2"][::-1], color=colours[::-1], edgecolor="white")
    ax.set_xlabel("Eta-squared (η²) — effect size", fontsize=11)
    ax.set_title(f"Top {len(top)} Most Discriminating Incoming Features", fontsize=13, fontweight="bold")
    ax.axvline(0.01, color="#E91E63", linestyle="--", linewidth=0.8, label="small effect (0.01)")
    ax.axvline(0.06, color="#FF9800", linestyle="--", linewidth=0.8, label="medium effect (0.06)")
    ax.axvline(0.14, color="#4CAF50", linestyle="--", linewidth=0.8, label="large effect (0.14)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_discrimination_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_boxplots(df, top_feats, out_dir, top_n=12):
    feats = top_feats[:top_n]
    if not feats: return
    n_cols = min(3, len(feats))
    n_rows = int(np.ceil(len(feats) / n_cols))
    cmap   = class_colours(df["label"].unique())
    palette= [cmap[c] for c in sorted(cmap)]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(feats):
        ax = axes[i]
        data_by_class = [df[df["label"] == c][feat].dropna().values
                         for c in sorted(df["label"].unique())]
        labels_list   = sorted(df["label"].unique())

        bp = ax.boxplot(data_by_class, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker=".", markersize=2, alpha=0.4))

        for patch, colour in zip(bp["boxes"], palette):
            patch.set_facecolor(colour)
            patch.set_alpha(0.85)

        ax.set_xticklabels(labels_list, rotation=25, ha="right", fontsize=8)
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Incoming Features Comparison — Boxplots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "boxplots_top12.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_violins(df, top_feats, out_dir, top_n=6):
    feats  = top_feats[:top_n]
    if not feats: return
    cmap   = class_colours(df["label"].unique())
    n_cols = min(2, len(feats))
    n_rows = int(np.ceil(len(feats) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(feats):
        ax = axes[i]
        data_records = []
        for cls in sorted(df["label"].unique()):
            for v in df[df["label"] == cls][feat].dropna().values:
                data_records.append({"class": cls, feat: v})
        sub = pd.DataFrame(data_records)

        classes  = sorted(sub["class"].unique())
        palette  = [cmap[c] for c in classes]

        sns.violinplot(data=sub, x="class", y=feat,
                       order=classes, palette=palette,
                       inner="box", linewidth=0.8, ax=ax)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Incoming Features — Violin Plots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "violin_top6.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap(df, feat_cols, out_dir, top_n=20):
    top_feats = feat_cols[:top_n]
    if not top_feats: return
    classes   = sorted(df["label"].unique())
    matrix    = np.zeros((len(classes), len(top_feats)))

    for ci, cls in enumerate(classes):
        sub = df[df["label"] == cls][top_feats]
        matrix[ci] = sub.mean().values

    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    denom   = col_max - col_min
    denom[denom == 0] = 1
    matrix_norm = (matrix - col_min) / denom

    cell_w = max(0.6, 12 / len(top_feats))
    fig, ax = plt.subplots(figsize=(len(top_feats) * cell_w + 2, len(classes) * 0.7 + 2))

    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Normalised mean (0=min, 1=max)")

    ax.set_xticks(range(len(top_feats)))
    ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_title(f"Normalised Incoming Means per Class (top {len(top_feats)})", fontsize=12, fontweight="bold")

    for ci in range(len(classes)):
        for fi in range(len(top_feats)):
            val = matrix[ci, fi]
            txt = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
            ax.text(fi, ci, txt, ha="center", va="center", fontsize=6.5,
                    color="black" if matrix_norm[ci, fi] < 0.6 else "white")

    plt.tight_layout()
    path = os.path.join(out_dir, "heatmap_mean_normalized.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pairplot(df, top_feats, out_dir, top_n=5):
    feats = top_feats[:top_n]
    if not feats: return
    cmap  = class_colours(df["label"].unique())
    sub   = df[feats + ["label"]].dropna()

    for f in feats:
        cap = sub[f].quantile(0.99)
        sub[f] = sub[f].clip(upper=cap)

    classes = sorted(sub["label"].unique())
    palette = [cmap[c] for c in classes]

    g = sns.pairplot(sub, hue="label", hue_order=classes,
                     palette=dict(zip(classes, palette)),
                     plot_kws=dict(alpha=0.4, s=10, linewidth=0),
                     diag_kind="kde", corner=True)
    g.figure.suptitle(f"Pairplot — Incoming top {len(feats)} features", y=1.01, fontsize=12, fontweight="bold")
    path = os.path.join(out_dir, "pairplot_top5.png")
    g.figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="PCAP multi-class feature comparison")
    p.add_argument("--input",  default="data/pcap_to_compare", help="Folder with .pcap files")
    p.add_argument("--output", default="reports/comparison",   help="Output folder for plots/CSVs")
    p.add_argument("--gran",   type=int, default=50,           help="Packet granularity (default 50)")
    return p.parse_args()


def derive_label(filename: str) -> str:
    """Dynamicznie mapuje pliki na osoby dla potrzeb analizy porównawczej."""
    filename_lower = filename.lower()
    if "kacper" in filename_lower:
        return "spotify_kacper"
    elif "agata" in filename_lower:
        return "spotify_agata"
    elif "martyna" in filename_lower:
        return "spotify_martyna"
    
    stem = Path(filename).stem
    return stem.split("_")[0].lower()


def main():
    args = parse_args()
    input_dir  = args.input
    output_dir = args.output
    gran       = args.gran

    os.makedirs(output_dir, exist_ok=True)

    pcap_files = sorted(Path(input_dir).glob("*.pcap"))
    if not pcap_files:
        print(f"[!] No .pcap files found in '{input_dir}'")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" PCAP Comparison Analysis (INCOMING FOCUS)")
    print(f"{'='*60}")

    all_dfs = []
    for pf in pcap_files:
        label = derive_label(pf.name)
        df    = pcap_to_dataframe(str(pf), label, gran)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("[!] No flow aggregates could be extracted.")
        sys.exit(1)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # 1. Zrównoważenie danych (Undersampling)
    print("\n[*] Balancing dataset to match the smallest class size...")
    min_samples = df_all["label"].value_counts().min()
    df_all = df_all.groupby("label").sample(n=min_samples, random_state=42).reset_index(drop=True)
    print(f"[+] Dataset balanced! New size: {len(df_all)} ({min_samples} per class)")

    # 2. Filtrowanie cech: Usuwamy 'fwd_' aby analizować tylko ruch PRZYCHODZĄCY (bwd)
    drop_cols = {"label", "actual_packets_in_flow", "agg_mode", "agg_value"}
    feat_cols = [c for c in df_all.columns if c not in drop_cols and df_all[c].dtype != object and not c.startswith("fwd_")]

    print(f"\n[*] Total flow aggregates : {len(df_all):,}")
    print(f"[*] Active Classes        : {sorted(df_all['label'].unique())}")
    print(f"[*] Analysed features     : {len(feat_cols)}")

    # 3. Obliczanie statystyk opisowych
    print("\n[*] Computing summary statistics...")
    summary_df = compute_summary(df_all, feat_cols)
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[+] Saved → {summary_path}")

    # 4. Analiza istotności cech (Kruskal-Wallis)
    print("[*] Computing discrimination scores (Kruskal-Wallis)...")
    disc_df = compute_discrimination(df_all, feat_cols)
    disc_path = os.path.join(output_dir, "feature_discrimination.csv")
    disc_df.to_csv(disc_path, index=False)
    print(f"[+] Saved → {disc_path}")

    top_feats = disc_df["feature"].tolist()

    if top_feats:
        print("\n  Top 10 most discriminating incoming features:")
        print(f"  {'Feature':<32} {'η²':>8}   {'p-value':>10}")
        print(f"  {'-'*55}")
        for _, row in disc_df.head(10).iterrows():
            effect = "●●●" if row["eta2"] >= 0.14 else ("●" if row["eta2"] >= 0.06 else "•")
            print(f"  {row['feature']:<32} {row['eta2']:>8.4f}   {row['p_value']:>10.2e}  {effect}")

    # 5. Generowanie nowoczesnych wykresów
    print("\n[*] Generating visualisations...")
    plot_class_distribution(df_all, output_dir)
    
    if not top_feats:
        print("[!] Brak cech do wygenerowania wykresów porównawczych.")
    else:
        plot_discrimination_bar(disc_df, output_dir, top_n=min(20, len(top_feats)))
        plot_boxplots(df_all, top_feats, output_dir, top_n=min(12, len(top_feats)))
        plot_violins(df_all, top_feats, output_dir, top_n=min(6, len(top_feats)))
        plot_heatmap(df_all, top_feats, output_dir, top_n=min(20, len(top_feats)))
        if len(top_feats) >= 2:
            plot_pairplot(df_all, top_feats, output_dir, top_n=min(5, len(top_feats)))

    print(f"\n{'='*60}\n Analysis complete! All outputs in: {output_dir}/\n{'='*60}")


if __name__ == "__main__":
    main()