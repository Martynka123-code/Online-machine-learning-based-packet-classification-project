import pandas as pd


def clean_incomplete_flows(csv_path, output_path=None):
    """
    Removes rows where:
    actual_packets_in_flow != agg_value

    Parameters
    ----------
    csv_path : str
        Input CSV dataset path

    output_path : str | None
        Output CSV path.
        If None -> overwrites original file
    """

    print(f"[*] Loading dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "actual_packets_in_flow",
        "agg_value"
    ]

    # Validate required columns
    for col in required_cols:
        if col not in df.columns:
            print(f"[!] Missing required column: {col}")
            return

    before = len(df)

    # Keep only complete packet windows
    df_clean = df[
        df["actual_packets_in_flow"] == df["agg_value"]
    ].copy()

    removed = before - len(df_clean)

    print(f"[*] Removed {removed} incomplete flows")
    print(f"[+] Remaining samples: {len(df_clean)}")

    # Overwrite original file if output_path not provided
    if output_path is None:
        output_path = csv_path

    df_clean.to_csv(output_path, index=False)

    print(f"[+] Cleaned dataset saved to: {output_path}")