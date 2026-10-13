# Data

The raw dataset is **not** committed to this repository. `data/raw/` and
`data/processed/` are git-ignored. The only committed file here besides this
README is `split_fingerprints.json`, which holds split counts and hashes of
review-ID lists and contains no review text.

## Source and provenance

| Field | Value |
|---|---|
| Dataset | *Wine Reviews*, Kaggle `zynicide/wine-reviews` (<https://www.kaggle.com/datasets/zynicide/wine-reviews>) |
| Creator | zackthoutt |
| Version | 4 (last updated 2017-11-27) |
| File used | `winemag-data-130k-v2.csv`. The dataset also contains `winemag-data_first150k.csv` (an earlier scrape without taster/title) and a JSON version; these are not used |
| Origin | Scraped from WineEnthusiast. Per the creator: first scrape June 2017, re-scraped 2017-11-22 to add `title`, `taster_name` and `taster_twitter_handle` |
| Licence | **CC BY-NC-SA 4.0**: attribution, non-commercial, share-alike |
| SHA-256 of the CSV | `52af2643c8ac29f010f0cc629dfbdda1c74aa0f332d11762af9ef3de4e567ac9` (pinned in `configs/default.toml`) |
| Size | 52,908,706 bytes (CSV); the download is a 17,180,972-byte zip |
| Retrieved | 2026-10-12 (metadata via the public Kaggle API) |

**Licence obligations for this project:**
- no commercial use;
- attribute the dataset when publishing results;
- share derived artifacts (e.g. trained models) under the same licence;
- do not redistribute the raw data.

## Acquisition

```bash
pip install -e ".[dev]"
winemood data download     # downloads, extracts and verifies the SHA-256
winemood data verify       # re-check the checksum at any time
```

No credentials are needed. The public Kaggle endpoint redirects to a signed
storage URL, and this worked on 2026-10-12. If it stops working:

- **Manual:** download `winemag-data-130k-v2.csv` from the Kaggle page into
  `data/raw/`, then run `winemood data verify`.
- **Kaggle CLI:** put your credentials in `~/.kaggle/kaggle.json`, never in
  this repository, then run

  ```bash
  kaggle datasets download -d zynicide/wine-reviews -f winemag-data-130k-v2.csv -p data/raw
  ```

  and unzip if necessary.

A checksum mismatch means the upstream file changed. Do not proceed without
investigating, because every published number is tied to the pinned file.

## Validation, splitting and EDA

```bash
winemood data validate          # -> reports/validation.json
winemood data split             # -> data/processed/splits/<name>/*.txt + data/split_fingerprints.json
winemood data split --check     # recompute and compare with the committed fingerprints
winemood data eda --publish     # -> reports/eda/ and docs/data-report.md + docs/figures/
```

The CSV's unnamed first column is exposed as a stable `review_id`.

The results and methodology are documented in:
- [`docs/data-report.md`](../docs/data-report.md): generated statistics;
- [`docs/experimental-protocol.md`](../docs/experimental-protocol.md): split
  protocol;
- [`docs/problem-formulation.md`](../docs/problem-formulation.md): targets.

## Known data issues (not altered)

- 10,016 exact duplicate descriptions. They are removed before splitting by
  keeping the lowest `review_id`. 23 duplicate groups carry conflicting
  scores.
- 2 descriptions contain mojibake (for example `crÃ¨me` for *crème*).
- `taster_name` is missing for 26,244 raw rows, and `price` for 8,996.
- Scores below 80 do not exist, because the publication does not review such
  wines.
