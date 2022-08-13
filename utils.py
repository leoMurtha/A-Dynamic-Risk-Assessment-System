import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

categorical_features = ["corporation"]

def preprocess_data(df, encoder):
    """
    Preprocess either test data or train data
    if test must pass the encoder
    """
    y = df["exited"]
    X = df.drop(["exited"], axis=1)
    
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if not encoder:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)
    else:
        X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_categorical, X_continuous], axis=1)

    return X, y, encoder