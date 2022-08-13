import pandas as pd
import numpy as np
import timeit
import os
import json
from joblib import load
from scipy.sparse import data
from utils import preprocess_data
import subprocess
import sys


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path']) 


##################Function to get model predictions
def model_predictions(dataset_path=None, dataset_name=None):
    #read the deployed model and a test dataset, calculate predictions
    model = load(os.path.join(model_path, "trainedmodel.pkl"))
    encoder = load(os.path.join(model_path, "encoder.pkl"))
    
    if dataset_name is None and dataset_path is None: 
        dataset_name = "testdata.csv"
        dataset_path = test_data_path
    
    print(os.path.join(dataset_path, dataset_name))
    df = pd.read_csv(os.path.join(dataset_path, dataset_name))

    X, y, _ = preprocess_data(df, encoder)

    y_pred = model.predict(X)

    return y_pred, y


##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    numeric = df.select_dtypes(include='int64')
    
    describe = numeric.iloc[:, :-1].agg(['mean', 'median', 'std'])

    return describe


##################Function to get summary statistics
def missing_data():
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    
    result = []
    for column in df.columns:
        count_null = df[column].isna().sum()
        count_not_null = df[column].count()
        count_total = count_not_null + count_null

        result.append([column, f"{str(int(count_null/count_total*100))}%"])
    
    return str(result)


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py

    result = []
    for procedure in ["training.py" , "ingestion.py"]:
        starttime = timeit.default_timer()
        os.system('python3 %s' % procedure)
        timing=timeit.default_timer() - starttime
        result.append([procedure, timing])
 
    return str(result)

##################Function to check dependencies
# Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated)


if __name__ == '__main__':
    model_predictions(None)
    print(execution_time())
    print(dataframe_summary())
    print(missing_data())
    print(outdated_packages_list())