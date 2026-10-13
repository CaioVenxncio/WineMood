# Data

The raw dataset is **not** committed to this repository. `data/raw/` and
`data/processed/` are git-ignored.

## Source

- **Dataset:** *Wine Reviews* by zynicide on Kaggle,
  <https://www.kaggle.com/datasets/zynicide/wine-reviews>
- **File used:** `winemag-data-130k-v2.csv`
- **Origin:** reviews scraped from WineEnthusiast.
- **Licence / terms:** *to be confirmed on the Kaggle page and recorded here in
  Phase 2, before any redistribution of data or derived artifacts.*

## Acquisition

Option A: download the dataset manually from the Kaggle page and place
`winemag-data-130k-v2.csv` in `data/raw/`.

Option B: use the Kaggle CLI (`pip install kaggle`):

```bash
# Credentials: put kaggle.json in ~/.kaggle/ (never in this repository).
kaggle datasets download -d zynicide/wine-reviews -p data/raw --unzip
```

The expected path is set in `configs/default.toml` (`[data] raw_csv`).

## Validation

```python
from winemood.config import load_config
from winemood.data.ingest import load_reviews
from winemood.data.schema import validate_reviews

cfg = load_config()
report = validate_reviews(load_reviews(cfg.data.raw_csv))
print(report)
```

Phase 2 will add a SHA-256 checksum of the file and a generated data report.
