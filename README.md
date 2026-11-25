# Project 3: Customer Segmentation, Retention Modeling, and Marketing Allocation

This project uses the anonymized dunnhumby Complete Journey dataset to identify household behavior groups, predict whether households return during a three-month holdout period, and recommend a $1,000,000 marketing allocation.

## Files

- `project3_analysis.R`: reproducible R analysis.
- `project3_report.qmd`: Quarto report source.
- `project3_report.html`: viewable report.
- `outputs/`: analysis tables.
- `figures/`: visualizations.
- `data/`: cached data used for this run.

## Git structure

The repository root contains the RStudio project, analysis script, Quarto source, rendered HTML report, README, and `.gitignore`. Analysis results are organized in `outputs/`, visualizations are organized in `figures/`, and optional local cached data are organized in `data/`. The `.gitignore` prevents local R session files and large cached data files from being committed unnecessarily.

## Reproduce

1. Open this folder as an RStudio Project.
2. Open `project3_analysis.R`.
3. Run the script from top to bottom. It installs missing packages, downloads the large transaction file with `completejourney::get_transactions()`, creates the household-level dataset, runs PCA, k-means, logistic regression, and writes the tables and figures.
4. Open `project3_report.qmd` and click **Render**.

The feature period is January 1–September 30, 2017. The holdout period is October 1–December 31, 2017. The unit of analysis is one household. `retained` equals 1 when the household makes at least one holdout-period purchase.

## Large-data management

The full transaction table is larger than the small sample distributed in the package. The code downloads it once, caches it locally, filters time periods before aggregation, and reduces transaction rows to one row per household. This avoids repeated downloads and limits the modeling table to 801 household records instead of carrying all transaction rows into PCA and clustering. The script handles the package/data import directly and documents the feature/holdout split. No synthetic customers or transactions were generated.

## Limitations

These are anonymized observational data. The observed holdout return rate is high, so specificity and identification of non-returners must be interpreted cautiously. Predicted return is not the same as incremental marketing response; a randomized treatment/control experiment is required to measure causation.
