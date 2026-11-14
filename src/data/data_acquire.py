from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "houses_for_rent_madrid.xlsx"
SHEET = "Houses_for_rent_madrid"

# Share of the most expensive rents removed before modelling (argsort on Rent, largest first).
TOP_RENT_FRACTION = 0.10


def load_df() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET)

    # Rows identical apart from Id are duplicate listings; keeping both would put the same flat in train and test folds.
    key = [c for c in df.columns if c != "Id"]
    df = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)

    # Drop the top TOP_RENT_FRACTION of rents: the luxury tail dominates euro errors and is not what this model is for.
    n_drop = int(np.ceil(len(df) * TOP_RENT_FRACTION))
    order = np.argsort(-df["Rent"].to_numpy(), kind="stable")
    keep = np.sort(order[n_drop:])
    return df.iloc[keep].reset_index(drop=True)


if __name__ == "__main__":
    df = load_df()
    print(df.head())
    print(f"\nShape: {df.shape}")
