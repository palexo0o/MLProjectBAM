from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src.data.data_acquire import load_df
except ModuleNotFoundError:
    from data_acquire import load_df


OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def prepare_ml_matrix(df_input: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Clean the dataset, encode categorical fields, standardize numeric features, and return ML-ready arrays."""
    work = df_input.copy()

    numeric_cols = [
        "extra_monthly_cost",
        "total_monthly_cost",
        "size_m2",
        "total_beds",
        "num_occupants",
        "online_booking",
        "has_insurance",
        "page_scraped",
        "house_number",
    ]
    categorical_cols = [
        "neighborhood",
        "gender_policy",
        "smoking_policy",
        "utilities_included",
        "city",
    ]

    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in categorical_cols:
        work[col] = work[col].fillna("unknown")

    y = work["price_eur"].astype(float).copy()
    feature_df = work.drop(columns=["listing_id", "property_type", "title", "url", "street", "extra_cost_note", "available_date", "price_eur"])

    X_num = feature_df[numeric_cols].copy()
    X_num = X_num.fillna(X_num.median())

    X_cat = pd.get_dummies(feature_df[categorical_cols], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.to_numpy(dtype=np.float64))
    y_array = y.to_numpy(dtype=np.float64)

    return X_scaled, y_array, X.columns.tolist()


if __name__ == "__main__":
    df = load_df()
    X_scaled, y, columns = prepare_ml_matrix(df)
    np.save(OUTPUT_DIR / "X_scaled.npy", X_scaled)
    np.save(OUTPUT_DIR / "y.npy", y)
    np.save(OUTPUT_DIR / "feature_names.npy", np.array(columns, dtype=object))

    print(f"Loaded dataset shape: {df.shape}")
    print(f"Scaled feature matrix shape: {X_scaled.shape}")
    print(f"Target vector shape: {y.shape}")
    print("Saved arrays:")
    print(OUTPUT_DIR / "X_scaled.npy")
    print(OUTPUT_DIR / "y.npy")
    print(OUTPUT_DIR / "feature_names.npy")
    print("\nFirst scaled sample:")
    print(X_scaled[0])
