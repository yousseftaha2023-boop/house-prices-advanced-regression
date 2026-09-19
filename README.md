# House Prices: Advanced Regression Pipeline

A leakage-aware linear regression pipeline for the Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition. The project focuses on careful data cleaning, feature engineering, and statistically grounded feature selection rather than on model complexity.

**Public leaderboard score:** ~0.169 (RMSE on log-transformed `SalePrice`)

## Overview

The script takes the raw Ames housing data through a full, reproducible workflow:

1. Exploration of both train and test sets (shape, duplicates, missing-value profile)
2. Domain-aware missing-value imputation
3. Feature engineering, outlier handling, skewness correction, and scaling
4. Rare-category merging with a custom function
5. Feature selection with RFECV
6. Linear regression with residual diagnostics on three evaluation splits
7. Submission file generation

## Pipeline

### Data splitting

The training data is split into three parts using `random_state=42`:

| Split | Share of data | Purpose |
|-------|---------------|---------|
| Train | ~79% | Fit imputers, scalers, encoders, and the model |
| Validation | ~9% | Tune and sanity-check |
| Hold-out | 12% | Final unbiased estimate |

All statistics used for imputation, clipping, scaling, and category merging are computed on the train split only and applied to the other splits and to the Kaggle test set.

### Missing values

- **Structural missings** (no pool, alley, fence, garage, basement, misc feature) are replaced with explicit categories such as `No Pool` or `No Garage`; `GarageYrBlt` becomes `0`.
- **`MasVnrType`** is set to `No Masonry` when `MasVnrArea` is `0`; otherwise the train mode is used.
- **`MasVnrArea`** and **`LotFrontage`** use the train median.
- **`Electrical`** uses the mode.
- **`FireplaceQu`** is imputed by sampling from the observed class distribution.
- Any remaining test-set missings fall back to the train median (numeric) or mode (categorical).

### Feature engineering

- `House_age_at_sale` (`YrSold - YearBuilt`)
- `DateSold` built from year and month; `MoSold` kept as a categorical feature to capture seasonality
- `BsmtHalfBath` converted to a binary `BsmtBath` indicator
- `PoolArea` dropped (redundant with `PoolQC`)
- Numeric features where at least 95% of values are zero are dropped
- Quality and condition ratings (`OverallQual`, `KitchenQual`, `OverallCond`, and similar) treated as categorical

### Outliers, skewness, and scaling

- Continuous features are winsorized at the 5th and 95th train percentiles (zero-inflated features are handled separately so that the zeros do not distort the IQR).
- Discrete features are winsorized at the 1st and 99th percentiles.
- Continuous features with |skewness| > 1 receive a `log1p` transform.
- Continuous and discrete features are scaled with `RobustScaler`; year features are left unscaled.
- The target is transformed with `log1p`, and predictions are converted back with `expm1`.

### Categorical encoding

`merge_rares_for_numerical_target` handles categorical columns using train-set statistics:

- Categories below a proportion threshold (5%) are merged into `Other`, with the merge rule taking the target median into account when only one rare class exists.
- Columns dominated by a single class (90% or more) are dropped.
- Categories unseen in train are mapped to `Other` in the other splits.

The result is then one-hot encoded with `pd.get_dummies(drop_first=True)`.

### Feature selection and modeling

- **Selection:** `RFECV` (3-fold CV, negative MSE) wrapped around `RidgeCV` over a log-spaced alpha grid, to cope with multicollinearity.
- **Model:** scikit-learn `LinearRegression` fitted on the selected features against `log1p(SalePrice)`.
- **Diagnostics** for train, validation, and hold-out sets: residuals vs. fitted values, Durbin-Watson statistic, Q-Q plots, and actual vs. predicted plots. RMSE is reported on the log scale.

## Getting started

### Prerequisites

- Python 3.9+
- scikit-learn 1.4+ (for `root_mean_squared_error`)

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

### Data

Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). The data is not included in this repository.

> [!IMPORTANT]
> The script currently reads and writes files using absolute Windows paths. Update the paths near the top of `advanced_housingprices_predection.py` (loading) and at the bottom (saving) to match your machine before running.

### Run

```bash
python advanced_housingprices_predection.py
```

The script prints the exploration and diagnostic output, shows the residual and Q-Q plots, and writes `submission_linear_reg.csv` with the columns `Id` and `SalePrice`, ready to upload to Kaggle.

## Project structure

```text
.
├── advanced_housingprices_predection.py   # Full pipeline: EDA, preprocessing, selection, model, submission
└── README.md
```

## Next steps

- Gradient boosting and stacked ensembles to close the gap with top leaderboard scores (roughly 0.11 to 0.13)
- Move preprocessing into a scikit-learn `Pipeline` / `ColumnTransformer` for cleaner cross-validation
- Compare the linear baseline against regularized models (Lasso, ElasticNet)
