from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "idealista_madrid_rooms_cleaned.csv"


def load_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":
    df = load_df()
    print(df.head())
    print(f"\nShape: {df.shape}")
