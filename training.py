"""
The module trains a Logistic Regression model
"""
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import json
from utils import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import fbeta_score, precision_score, recall_score
import joblib

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)

    return precision, recall, fbeta


# Function for training the model
def train_model():
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    X, y, encoder = preprocess_data(df, None)
    
    #80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.20)
    

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # Split the data into features and label
    
    model.fit(X_train, y_train)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    cv_report = cross_val_score(model, X_train, y_train,
                                cv=cv, n_jobs=-1)

    print('KFold CV report.')
    print(cv_report)
    
    y_pred = model.predict(X_test)
    
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    print('Test set/Holdout Set metrics: \n')
    print(f'precision: {precision}')
    print(f'\nrecall: {recall}')
    print(f'fbeta: {fbeta}\n')

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    joblib.dump(model, os.path.join(model_path, 'trainedmodel.pkl'))
    joblib.dump(encoder, os.path.join(model_path, 'encoder.pkl'))
    


if __name__ == '__main__':
    train_model()
    print("Model trained and saved along with encoder")