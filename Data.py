import io 
import requests
from pathlib import Path 
import zipfile

import pandas as pd
 
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"

def download_movielens() -> Path:
    data_file = DATA_DIR / "ml-100k"
    if data_file.exists():
        return data_file 
    
    DATA_DIR.mkdir(exist_ok=True)
    response = requests.get(URL)
    response.raise_for_status()

    zipfile.ZipFile(io.BytesIO(response.content)).extractall(DATA_DIR)
    return data_file

def load_ratings() -> pd.DataFrame:
    path = download_movielens() / "u.data"
    df = pd.read_csv(path, sep="\t", names=["user", "item", "ratings", "timestamp"])

    df["user"] -= 1
    df["item"] -= 1
    return df

def leave_one_out_split(df: pd.DataFrame):
    df = df.sort_values(["user", "timestamp"])
    test = df.groupby("user").tail(1)
    train = df.drop(test.index)
    print(train)
    return train.reset_index(drop=True), test.reset_index(drop=True)

if __name__ == "__main__":
    leave_one_out_split(load_ratings())


