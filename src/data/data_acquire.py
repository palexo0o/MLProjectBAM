import pandas as pd


def load_df() -> pd.DataFrame:
    return pd.read_csv("/Users/philipalexopoulos/MLProjectBAM/src/data/idealista_madrid_rooms_cleaned.csv")


df = load_df()


if __name__ == "__main__":
    print(df.head())
    print(f"\nShape: {df.shape}")