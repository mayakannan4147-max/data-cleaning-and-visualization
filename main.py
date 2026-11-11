"""Clean a CSV dataset and generate a static visual report."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _iqr_bounds(series: pd.Series) -> tuple[float, float] | None:
    values = series.dropna()
    if values.empty:
        return None

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    spread = q3 - q1
    if pd.isna(spread) or spread == 0:
        return None
    return float(q1 - 1.5 * spread), float(q3 + 1.5 * spread)


def clean_data(
    source: pd.DataFrame, *, cap_outliers: bool = False
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a cleaned copy and a summary of the applied cleaning steps."""
    data = source.copy()
    original_rows = len(data)

    renamed_columns = [str(column).strip() for column in data.columns]
    if len(set(renamed_columns)) != len(renamed_columns):
        raise ValueError("Column names must be unique after trimming whitespace.")
    data.columns = renamed_columns

    missing_before = data.isna().sum()
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    numeric_columns = data.select_dtypes(include="number").columns
    outlier_counts: dict[str, int] = {}
    outliers_capped: dict[str, int] = {}
    for column in numeric_columns:
        bounds = _iqr_bounds(data[column])
        if bounds is None:
            continue
        lower, upper = bounds
        outlier_mask = data[column].notna() & (
            (data[column] < lower) | (data[column] > upper)
        )
        count = int(outlier_mask.sum())
        if count:
            outlier_counts[column] = count
            if cap_outliers:
                data[column] = data[column].clip(lower=lower, upper=upper)
                outliers_capped[column] = count

    imputation: dict[str, str] = {}
    for column in data.columns:
        missing_count = int(data[column].isna().sum())
        if missing_count == 0:
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            median = data[column].median()
            if pd.notna(median):
                data[column] = data[column].fillna(median)
                imputation[column] = "median"
        else:
            modes = data[column].mode(dropna=True)
            if not modes.empty:
                data[column] = data[column].fillna(modes.iloc[0])
                imputation[column] = "mode"

    summary: dict[str, object] = {
        "original_rows": original_rows,
        "cleaned_rows": len(data),
        "duplicate_rows_removed": duplicate_count,
        "missing_before": missing_before[missing_before > 0].to_dict(),
        "missing_after": data.isna().sum()[data.isna().sum() > 0].to_dict(),
        "imputation": imputation,
        "outlier_counts": outlier_counts,
        "outliers_capped": outliers_capped,
        "cap_outliers": cap_outliers,
    }
    return data, summary


def create_visualization(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    summary: dict[str, object],
    image_path: Path,
) -> None:
    numeric_columns = cleaned.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column
        for column in cleaned.columns
        if pd.api.types.is_string_dtype(cleaned[column].dtype)
        or isinstance(cleaned[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(cleaned[column].dtype)
    ]

    panels = [("numeric", column) for column in numeric_columns[:4]]
    panels.extend(("categorical", column) for column in categorical_columns[:4])
    if not panels:
        panels = [("empty", "")]

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(11, max(4, 3.2 * len(panels))),
        squeeze=False,
    )
    for axis, (kind, column) in zip(axes[:, 0], panels):
        if kind == "numeric":
            cleaned[column].dropna().hist(ax=axis, bins=20, edgecolor="white")
            axis.set_title(f"Numeric distribution: {column}")
            axis.set_xlabel("Value")
            axis.set_ylabel("Count")
        elif kind == "categorical":
            cleaned[column].astype(str).value_counts().head(10).sort_values().plot.barh(
                ax=axis, color="#3b82f6"
            )
            axis.set_title(f"Most common values: {column}")
            axis.set_xlabel("Count")
        else:
            axis.text(0.5, 0.5, "No plottable columns found", ha="center", va="center")
            axis.set_axis_off()

    fig.suptitle(
        f"Dataset overview | {summary['original_rows']} rows → "
        f"{summary['cleaned_rows']} cleaned rows",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _table_rows(values: dict[object, object], empty_message: str) -> str:
    if not values:
        return f"<tr><td colspan='2'>{html.escape(empty_message)}</td></tr>"
    return "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in values.items()
    )


def create_html_report(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    summary: dict[str, object],
    image_name: str,
    report_path: Path,
) -> None:
    missing = summary["missing_before"]
    outliers = summary["outlier_counts"]
    imputation = summary["imputation"]
    missing_after = summary["missing_after"]
    image_src = html.escape(image_name, quote=True)
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Data cleaning report</title>
  <style>
    body {{ max-width: 960px; margin: 2rem auto; padding: 0 1rem; font: 16px/1.5 system-ui, sans-serif; color: #172033; }}
    h1, h2 {{ color: #15345b; }} .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
    .card {{ background: #f1f5f9; border-radius: 8px; padding: 1rem; min-width: 150px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border-bottom: 1px solid #dbe3ec; padding: .55rem; text-align: left; }}
    th {{ background: #f1f5f9; }} img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Data cleaning report</h1>
  <p>Review the rules below before using this output for decisions. Outliers are
     reported but left unchanged unless the run used <code>--cap-outliers</code>.</p>
  <div class="cards">
    <div class="card"><strong>Rows</strong><br>{len(original):,} → {len(cleaned):,}</div>
    <div class="card"><strong>Columns</strong><br>{len(original.columns):,}</div>
    <div class="card"><strong>Duplicate rows removed</strong><br>{summary['duplicate_rows_removed']:,}</div>
    <div class="card"><strong>Outlier strategy</strong><br>{"IQR capping" if summary["cap_outliers"] else "Report only"}</div>
  </div>
  <h2>Visual overview</h2>
  <img src="{image_src}" alt="Numeric distributions and common category values">
  <h2>Missing values before cleaning</h2>
  <table><thead><tr><th>Column</th><th>Missing values</th></tr></thead>
    <tbody>{_table_rows(missing, "No missing values found.")}</tbody></table>
  <h2>Imputation applied</h2>
  <table><thead><tr><th>Column</th><th>Method</th></tr></thead>
    <tbody>{_table_rows(imputation, "No imputation needed.")}</tbody></table>
  <h2>Potential outliers (IQR rule)</h2>
  <table><thead><tr><th>Column</th><th>Values outside 1.5×IQR</th></tr></thead>
    <tbody>{_table_rows(outliers, "No potential outliers detected.")}</tbody></table>
  <p>Rows remaining with missing values: {sum(missing_after.values()):,}.
     All-null columns cannot be imputed automatically. Inspect the cleaned CSV
     and decide whether those columns should be retained.</p>
</body>
</html>
"""
    report_path.write_text(report, encoding="utf-8")


def process_csv(
    input_path: Path, output_dir: Path, *, cap_outliers: bool = False
) -> tuple[Path, Path, Path]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")

    original = pd.read_csv(input_path, encoding="utf-8-sig")
    cleaned, summary = clean_data(original, cap_outliers=cap_outliers)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "cleaned_data.csv"
    image_path = output_dir / "visualization.png"
    report_path = output_dir / "cleaning_report.html"
    cleaned.to_csv(cleaned_path, index=False)
    create_visualization(original, cleaned, summary, image_path)
    create_html_report(original, cleaned, summary, image_path.name, report_path)
    return cleaned_path, report_path, image_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean a CSV file and generate a visual HTML report."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the raw CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for cleaned data and report (default: output)",
    )
    parser.add_argument(
        "--cap-outliers",
        action="store_true",
        help="Cap numeric outliers at 1.5×IQR bounds; default is report only",
    )
    args = parser.parse_args()

    cleaned_path, report_path, image_path = process_csv(
        args.input_csv, args.output_dir, cap_outliers=args.cap_outliers
    )
    print(f"Cleaned CSV: {cleaned_path}")
    print(f"HTML report: {report_path}")
    print(f"Visualization: {image_path}")


if __name__ == "__main__":
    main()
