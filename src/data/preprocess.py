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


# Monthly rent in euros for a whole flat.
TARGET = "Rent"

# Columns that are the target or an identifier / free text.
EXCLUDED = ["Id", "Address", "Number", TARGET]

NUMERIC_COLS = [
    "Bedrooms",
    "Sq.Mt",       # complete in this dataset
    "Floor",       # -1 = basement; implausible values (outside -1..MAX_FLOOR) are set to missing in prepare_xy
    "Outer",       # 1 = exterior-facing
    "Elevator",
    "Penthouse",   # 0/1 property-type flags
    "Cottage",
    "Duplex",
    "Semidetached",
]

# Area is the neighbourhood (140 levels); District is the wider district (20 levels).
CATEGORICAL_COLS = [
    "Area",
    "District",
]
MIN_FREQ = 10
MAX_FLOOR = 30


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
    work["Floor"] = work["Floor"].where(work["Floor"].between(-1, MAX_FLOOR))
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
