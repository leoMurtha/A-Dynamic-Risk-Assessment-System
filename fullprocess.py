from hashlib import new
from training import train_model
from deployment import store_model_into_pickle
from diagnostics import model_predictions
from sklearn.metrics import f1_score
from ingestion import merge_multiple_dataframe
import json
import os

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
prod_folder_path = os.path.join(config['prod_deployment_path'])
output_folder_path = os.path.join(config['output_folder_path'])
artifacts_path = os.path.join(config['output_model_path'])


filenames = os.listdir(prod_folder_path)

# Check and read new data first, read ingestedfiles.txt
ingested_files = []
with open(os.path.join(prod_folder_path, 'ingestedfiles.txt')) as file:
    ingested_files = file.read().splitlines()


# Determine whether the source data folder has files that aren't listed in
# ingestedfiles.txt
source_files = os.listdir(input_folder_path)

continue_pipeline = False
for file in source_files:
    if file not in ingested_files:
        print('There are new files to ingest')
        continue_pipeline = True
        break
    
if not continue_pipeline:
    print("No new files found")
    exit(0)

# Ingest new data
merge_multiple_dataframe(input_folder_path, output_folder_path)

# Checking for model drift
with open(os.path.join(prod_folder_path, 'latestscore.txt')) as file:
    old_f1_score = float(file.read())

pred, y_test = model_predictions(output_folder_path, 'finaldata.csv')
new_f1_score = float(f1_score(pred, y_test))

# Check whether the score from the deployed model is different from the
# score from the model that uses the newest ingested data
if new_f1_score >= old_f1_score:
    print("No drift in model detected")
    print("F1 Scores are greater or equal")
    print(f"Old F1: {old_f1_score}  New F1: {new_f1_score}")
    exit(0)
else:
    # If there is model drift, retrain
    print("Retraining and redeploying the model because of model drift")
    train_model()

# Re-deploy if there is evidence for model drift, re-run the deployment.py
# script
store_model_into_pickle(artifacts_path)

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
os.system("python3 diagnostics.py")
os.system("python3 reporting.py")
os.system("python3 apicalls.py")
