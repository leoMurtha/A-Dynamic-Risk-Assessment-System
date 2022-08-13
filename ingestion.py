import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from glob import glob




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

# Read all files from a data folder specified in config.son, without manually writing file
# names in your script.
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



def merge_multiple_dataframe(input_folder_path, output_folder_path):
    """
    Reads multiples files of type csv from input_folder_path
    and saves into output_folder_path as format csv
    """
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    
    #check for datasets, compile them together, and write to an output file
    files = glob(f"{input_folder_path}/*.csv")
    
    # Compile the files you read into a single pandas DataFrame.
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    
    # De-dupe the pandas DataFrame you compiled.
    df = df.drop_duplicates()

    # Write the pandas data frame to your workspace as finaldata.csv.
    df.to_csv(f"{output_folder_path}/finaldata.csv" , index=False)

    # Save a record of all the files you have read in ingestedfiles.txt.
    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as report_file:
        for line in files:
            report_file.write(line + "\n")


if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path, output_folder_path)
