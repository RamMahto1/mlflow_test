# utils/data_loader.py

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(test_size=0.2, random_state=42):
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    X = df.drop('target', axis=1)
    y = df['target']

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
