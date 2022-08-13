import requests
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

# Call each API endpoint and store the responses
response1 = requests.post(f"{URL}/prediction",
                          json={"filename": None, "filepath": None}).text
response2 = requests.get(f"{URL}/scoring").text
response3 = requests.get(f"{URL}/summarystats").text
response4 = requests.get(f"{URL}/diagnostics").text

# Combine all API responses
responses = "\n".join([response1, response2, response3, response4])

# Save the response
with open('config.json', 'r') as file:
    config = json.load(file)
    output_model_path = os.path.join(config['output_model_path'])

if not os.path.exists(output_model_path):
        os.mkdir(output_model_path)
        
with open(os.path.join(output_model_path, "apireturns.txt"), "w") as file:
    file.write(responses)