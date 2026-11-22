"""
Step 2 - student-alone baseline (the bottom rung of the comparison ladder).

Embeds proteins with ESM-2-8M (the student model, frozen), trains a one-vs-rest
logistic regression on folds 0-3 and tests on the last fold.
This is NOT distillation yet: it is the lower bound distilled students must beat.

Usage:  python src/baseline_student.py --n 3000      (GPU recommended)
Outputs: results/baseline_metrics.txt, data/esm2_8m_embeddings.npz (cache)
"""
import argparse, os
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

MODEL = "facebook/esm2_t6_8M_UR50D"
LABELS = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion",
          "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus",
          "Peroxisome"]


def embed(seqs, max_len, batch_size, device):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = tok([s[:max_len] for s in seqs[i:i + batch_size]],
                        return_tensors="pt", padding=True).to(device)
            h = model(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1)
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())   # mean pooling
            if (i // batch_size) % 25 == 0:
                print(f"  embedded {i}/{len(seqs)}")
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/deeploc_clean.csv")
    ap.add_argument("--n", type=int, default=3000, help="proteins to use (0 = all)")
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(args.csv)
    labels = [c for c in LABELS if c in df.columns]
    if args.n and args.n < len(df):
        df = df.sample(n=args.n, random_state=42).reset_index(drop=True)

    cache = f"data/esm2_8m_embeddings_{len(df)}.npz"
    if os.path.exists(cache):
        X = np.load(cache)["X"]
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding {len(df)} proteins with ESM-2-8M on {device} ...")
        X = embed(df["Sequence"].tolist(), args.max_len, args.batch_size, device)
        np.savez_compressed(cache, X=X)
    Y = df[labels].values.astype(int)

    if "Partition" in df.columns:
        test = (df["Partition"] == df["Partition"].max()).values
        split = f"train folds != {df['Partition'].max()}, test fold {df['Partition'].max()}"
    else:
        test = np.random.RandomState(0).rand(len(df)) < 0.2
        split = "random 80/20 split (no Partition column)"

    keep = Y[~test].sum(0) > 0                    # labels with at least one training example
    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
    clf.fit(X[~test], Y[~test][:, keep])
    P = np.zeros_like(Y[test]); P[:, keep] = clf.predict(X[test])
    T = Y[test]

    acc = accuracy_score(T, P)
    f1 = f1_score(T, P, average="macro", zero_division=0)
    mcc = np.mean([matthews_corrcoef(T[:, j], P[:, j]) for j in range(T.shape[1])])
    per = f1_score(T, P, average=None, zero_division=0)

    lines = [
        "Student-alone baseline (PRELIMINARY)",
        f"Model: ESM-2-8M frozen embeddings + logistic regression",
        f"Proteins: {len(df)} ({(~test).sum()} train / {test.sum()} test), {split}",
        f"Exact-match accuracy: {acc:.3f}",
        f"Macro-F1:             {f1:.3f}",
        f"Mean per-label MCC:   {mcc:.3f}",
        "", "F1 per compartment:",
    ] + [f"  {l:<24}{v:.3f}" for l, v in zip(labels, per)]
    report = "\n".join(lines)
    print("\n" + report)
    open("results/baseline_metrics.txt", "w").write(report + "\n")


if __name__ == "__main__":
    main()
