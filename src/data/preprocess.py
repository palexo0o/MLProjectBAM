from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from src.data.data_acquire import load_df
except ModuleNotFoundError:
    from data_acquire import load_df


# What renters actually pay per month. Predicting this (not price_eur) matches the consumer question.
TARGET = "total_monthly_cost"

# Columns that are the target, a component of it, or identifiers / free text.
# total_monthly_cost == price_eur + extra_monthly_cost exactly, so using either as a feature leaks the target.
EXCLUDED = [
    "listing_id",
    "property_type",
    "title",
    "url",
    "street",
    "house_number",
    "extra_cost_note",
    "available_date",
    "price_eur",
    "extra_monthly_cost",
    TARGET,
]

NUMERIC_COLS = [
    "size_m2",          # ~88% missing: imputed, plus a missing-indicator column
    "total_beds",       # likely flat-level capacity, not per-room; verify against listing text
    "num_occupants",
    "online_booking",
    "has_insurance",
    "page_scraped",
]

# Neighborhood has 112 levels, median 7 listings each. Levels with fewer than MIN_FREQ rows are pooled.
CATEGORICAL_COLS = [
    "neighborhood",
    "gender_policy",
    "smoking_policy",
    "utilities_included",
]
MIN_FREQ = 10


def build_preprocessor() -> ColumnTransformer:
    """Imputation and encoding only. No scaling here: scaling is model-specific, and every step is fit inside each CV fold."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_FREQ, sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, CATEGORICAL_COLS),
        ]
    )


def prepare_xy(df_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the raw feature frame and the target. Transformation happens inside a fitted pipeline, not here."""
    work = df_input.copy()
    for col in NUMERIC_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    X = work[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = pd.to_numeric(work[TARGET], errors="coerce").astype(float)
    return X, y


if __name__ == "__main__":
    df = load_df()
    X, y = prepare_xy(df)
    Xt = build_preprocessor().fit_transform(X)
    print(f"Loaded dataset shape: {df.shape}")
    print(f"Feature frame shape (pre-encoding): {X.shape}")
    print(f"Encoded matrix shape (fit on full data, for inspection only): {Xt.shape}")
    print(f"Target: {TARGET}, mean {y.mean():.1f}, min {y.min():.0f}, max {y.max():.0f}")
