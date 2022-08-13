from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import json
import os
from diagnostics import model_predictions


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path'])


# Function for reporting
def score_model():
    
    y_pred, y_test = model_predictions(dataset_path=None)
    
    print(y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fig = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    if not os.path.exists(output_model_path):
        os.mkdir(output_model_path)
        
    plt.figure(figsize=(10, 8))
    fig.plot()
    plt.savefig(os.path.join(output_model_path, "confusionmatrix.png"))


if __name__ == '__main__':
    score_model()