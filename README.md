# Data Cleaning & Visualization

A reusable command-line workflow for cleaning a raw CSV file and generating a
visual report. The repository does not include a dataset; use your own CSV.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py path\to\raw_data.csv
```

The default output directory is `output`. It contains:

- `cleaned_data.csv` — rows with exact duplicates removed and missing values
  imputed where a median or mode can be calculated.
- `cleaning_report.html` — row counts, missing-value and imputation details,
  potential outlier counts, and the visualization.
- `visualization.png` — distributions for up to four numeric columns and the
  most common values for up to four categorical columns.

To cap numeric outliers at the 1.5×IQR bounds, opt in explicitly:

```powershell
python main.py path\to\raw_data.csv --cap-outliers
```

By default, potential outliers are reported and not changed. This avoids
silently altering valid extreme observations. Numeric missing values are filled
with the column median; other missing values are filled with the most frequent
value. An all-null column cannot be imputed automatically and remains missing.

## Test

```powershell
python -m unittest discover -s tests -v
```
