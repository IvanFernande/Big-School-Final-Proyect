import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import SEED, DATA_RAW, DATA_PROCESSED


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)
    df = df.rename(columns={"H1": "id"})
    df["Tags"] = df["Tags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    df["Body"] = df["Body"].fillna("")
    df["Department"] = df["Department"].fillna("Unknown").str.strip()
    df["Priority"] = df["Priority"].str.strip()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Body"] = df["Body"].fillna("").astype(str)
    df["n_tags"] = df["Tags"].apply(len)
    df["len_words"] = df["Body"].apply(lambda t: len(str(t).split()))
    return df


def split(df: pd.DataFrame):
    X = df[["Body", "Department", "n_tags", "len_words"]]
    y = df["Priority"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def main():
    df = add_features(load())
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)
    print(df.head())


if __name__ == "__main__":
    main()
