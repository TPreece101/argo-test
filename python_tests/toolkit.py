import io
import requests
import zipfile

import pandas as pd

def download_data():
    try:
        # Specify url
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
        # Make request
        req = requests.get(url)
        # Extract zip file
        z = zipfile.ZipFile(io.BytesIO(req.content))
        z.extractall()
    except:
        return False
    finally:
        return True

def find_numeric_cat_cols(data_frame:pd.DataFrame):
    num_cols = []
    cat_cols = []
    
    for i in range(len(data_frame.dtypes)):
        if data_frame.dtypes[i] == 'O':
            cat_cols.append(data_frame.dtypes.index[i])
        else:
            num_cols.append(data_frame.dtypes.index[i])
            
    return(num_cols, cat_cols)