"""
Step 1 - data preparation.

Downloads the DeepLoc 2.0 SwissProt dataset, checks it, prints statistics,
saves two charts and a cleaned CSV that the other scripts use.

Usage:  python src/prepare_data.py            (downloads if needed)
        python src/prepare_data.py --csv path/to/your.csv
Outputs (in results/ and data/):
  data/deeploc_clean.csv
  results/data_stats.txt
  results/milestone_label_counts.png
  results/milestone_lengths.png
"""
import argparse, os, urllib.request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

URL = ("https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/"
       "Swissprot_Train_Validation_dataset.csv")
LABELS = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion",
          "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus",
          "Peroxisome"]
MAX_LEN = 1000


def load(csv_path):
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        print(f"Downloading DeepLoc 2.0 to {csv_path} ...")
        try:
            urllib.request.urlretrieve(URL, csv_path)
        except Exception as e:
            raise SystemExit(f"Download failed ({e}). Download the SwissProt CSV from the "
                             f"DeepLoc 2.0 website and pass it with --csv.")
    return pd.read_csv(csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/deeploc_raw.csv")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    df = load(args.csv)
    labels = [c for c in LABELS if c in df.columns]
    missing = sorted(set(LABELS) - set(labels))
    if "Sequence" not in df.columns or len(labels) < 8:
        raise SystemExit(f"Unexpected columns: {list(df.columns)}")
    fold = "Partition" if "Partition" in df.columns else None

    # clean: drop empty / duplicate sequences, keep only valid amino-acid letters
    n0 = len(df)
    df = df.dropna(subset=["Sequence"])
    df["Sequence"] = df["Sequence"].str.upper().str.replace(r"[^A-Z]", "", regex=True)
    df = df[df["Sequence"].str.len() > 0].drop_duplicates(subset="Sequence")
    df[labels] = df[labels].fillna(0).astype(int)
    df = df[df[labels].sum(axis=1) > 0]          # every protein needs >= 1 location

    counts = df[labels].sum().sort_values(ascending=False)
    lengths = df["Sequence"].str.len()
    multi = (df[labels].sum(axis=1) > 1).mean()

    lines = [
        f"Raw rows: {n0}   after cleaning: {len(df)}",
        f"Compartments used: {len(labels)}" + (f" (missing: {missing})" if missing else ""),
        f"Proteins with more than one location: {multi:.1%}",
        f"Sequence length: median {lengths.median():.0f}, mean {lengths.mean():.0f}, "
        f"max {lengths.max()}; {(lengths > MAX_LEN).mean():.1%} longer than {MAX_LEN}",
        "", "Proteins per compartment:", counts.to_string(),
    ]
    if fold:
        lines += ["", "Proteins per fold:", df[fold].value_counts().sort_index().to_string()]
    report = "\n".join(lines)
    print(report)
    open("results/data_stats.txt", "w").write(report + "\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot.barh(ax=ax, color="#0F4C5C")
    ax.invert_yaxis()
    ax.set_xlabel("Number of proteins")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig("results/milestone_label_counts.png", dpi=200)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(lengths.clip(upper=2500), bins=60, color="#0F4C5C")
    ax.axvline(MAX_LEN, color="#E36414", ls="--", label=f"truncated at {MAX_LEN}")
    ax.set_xlabel("Sequence length (amino acids)"); ax.set_ylabel("Proteins"); ax.legend()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig("results/milestone_lengths.png", dpi=200)

    keep = ["Sequence"] + labels + ([fold] if fold else []) + \
           [c for c in ("ACC", "Kingdom") if c in df.columns]
    df[keep].to_csv("data/deeploc_clean.csv", index=False)
    print("\nSaved data/deeploc_clean.csv and charts in results/")


if __name__ == "__main__":
    main()
