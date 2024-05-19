from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np


def Load_data(path_dataset):
    file_path = Path(path_dataset)
    if not file_path.exists():
        Path("dataset").mkdir(parents=True, exist_ok=True)
        # urllib.request.urlretrieve(url, file_path)

    return pd.read_csv(file_path, )


def Get_conflicted_data(dataset, target, exclude):
    data = dataset.drop(exclude, axis=1)
    return np.where((dataset.iloc[:, 1:-1].nunique(axis=1,) != 1), "Yes", "No")
