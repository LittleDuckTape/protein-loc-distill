# Distributed Multi-Teacher Knowledge Distillation for Protein Subcellular Localization

Final project for Computational Biology, Artificial Intelligence, Distributed Systems
and Research Methodology (BINUS, Odd Semester 2026/2027).
Irene Angelina · Farrell Raffelino Sunarman

Two large protein language models (ESM-2-650M and ProtT5-XL) act as teachers on
separate worker nodes. A small student (ESM-2-8M) learns from the true labels and
from the teachers' confidence-weighted soft targets, then runs inside an educational
web app.

## Status (milestone, Week 10)

| Part | Status |
|---|---|
| Data preparation (`src/prepare_data.py`) | Partially done: DeepLoc 2.0 loaded, cleaned, statistics. HPA test set not yet |
| Student baseline (`src/baseline_student.py`) | Partially done: student alone, no distillation yet |
| Containerisation (`Dockerfile`) | Partially done: one image for all nodes. Coordinator and workers not yet |
| Teacher heads, distributed serving, distillation, web app, tests | Not started |

## Run

```bash
pip install -r requirements.txt
python src/prepare_data.py          # downloads DeepLoc 2.0, writes results/ and data/
python src/baseline_student.py      # ESM-2-8M baseline on 3000 proteins (GPU recommended)
```

On Google Colab or Kaggle: upload the repo (or `git clone` it), turn the GPU on, and run
the same two commands with `!` in front.

If the DeepLoc download fails, download the SwissProt CSV from the DeepLoc 2.0 website and run
`python src/prepare_data.py --csv path/to/file.csv`.

### Docker

```bash
docker build -t protein-loc .
docker run --rm -v "$PWD/results:/app/results" -v "$PWD/data:/app/data" protein-loc
```

## Layout

```
src/prepare_data.py       data download, cleaning, statistics, charts
src/baseline_student.py   ESM-2-8M embeddings + logistic regression baseline
data/                     datasets and cached embeddings (not committed)
results/                  charts and metrics
Dockerfile                same environment on every node
```

## Data and models

- DeepLoc 2.0 SwissProt dataset (Thumuluri et al., 2022), 5 homology-partitioned folds
- ESM-2 (Lin et al., 2023) via Hugging Face `facebook/esm2_t6_8M_UR50D`
