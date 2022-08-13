from flask import Flask, request
import joblib
import json
import os
from diagnostics import model_predictions, dataframe_summary
from diagnostics import missing_data, execution_time, outdated_packages_list
from scoring import score_model


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])



with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), "rb") as model:
    prediction_model = joblib.load(model)


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    filepath = request.json.get('filepath')
    filename = request.json.get('filename')
    pred, _ = model_predictions(dataset_path=filepath, dataset_name=filename)
    return str(pred)


# Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    score = score_model()

    return str(score)


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    stats = dataframe_summary()
    return str(stats)


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():

    time = execution_time()
    null_values = missing_data()
    outdated = outdated_packages_list()

    return str(f"execution_time: {time} \n missing_values: {null_values} \n outdated_packages: {outdated}")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)