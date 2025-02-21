import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.cluster import *
from sklearn.preprocessing import *
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import preprocessing
from glob import glob
from sklearn.model_selection import train_test_split
import warnings
import requests
import pandas as pd
import datetime
import time
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
#from xgboost import XGBRegressor
from sources.src.upload_data import load_data
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import *
from datetime import datetime
import joblib
from sklearn.metrics import *
from lightgbm import LGBMRegressor 
from xgboost import XGBRegressor


def main_predict(pattern_ranges = None, servername="intelift", wellid=4200, tags=None, target_column="504"):
    # rename variables
    inputs = tags
    output = target_column 

     
    # load data
    all_tags = inputs + [output] 


    #-------> here we start the update 
    dfs = []
    try:
        for time_range in pattern_ranges:
            start, end, well_id = time_range[0], time_range[1], time_range[2]
            df_p = load_data(servername=str(servername), start=int(start), end=int(end), tags=list(all_tags), wellid=int(well_id))
            dfs.append(df_p)
            print(f"Loaded data for well ID: {well_id}")
        
        df_all = pd.concat(dfs, ignore_index=True)
        print(f"------------------------- Data loaded successfully and has a length of {df_all.shape} -------------------------")
    except Exception as e:
        print("Error in search logic function for data loading")
        print(e)

    # processig Time to Index
    df = df_all.set_index('0')

    #------->  here we finish the update


    #df = load_data(servername = servername, start=start, end=end, tags=all_tags, wellid=wellid)
    #print("data is loaded and has a shape of ", df.shape)

    # Time to Index
    #df = df.set_index('0')
    
    # preprocess data to include the frequency with a limited values
    df  = preprocess(df = df)

    try:
        train_df,test_df, scaler,y_scaler_value = split_scale(df = df, y=target_column, inputs=inputs)
        print("y_scaler value", y_scaler_value)
    except Exception as e:
        print(e)
        print("Error in split scale function")

    X = train_df[inputs].values
    Y = train_df[output]
    
    X_test = test_df[inputs].values
    Y_test = test_df[output]
    

    models = {
        'Linear Regression': LinearRegression(),
        'LGBMRegressor': LGBMRegressor(metric='rmse',verbose=-1), # the most of time, this algorithm gives the best results
        'Ridge Regression': Ridge(),
        'Decision Tree': DecisionTreeRegressor(),
        'XGBRegressor': XGBRegressor(objective ='reg:squarederror', n_estimators=100, learning_rate=0.1)
    }

    best_performance = float('-inf')
    best_model = None
    best_mse = float('-inf')
    model_name = ''

    # Initialize dictionaries to store results and predictions
    res = {}

    for name, model in models.items():
        print(name, "start training")
        model.fit(X, Y)
        predictions = model.predict(X_test)
        performance = round(r2_score(Y_test, predictions) * 100, 2)

        r2 = r2_score(Y_test, predictions) *100 
        mse = np.sum((np.abs(Y_test-predictions)/len(Y_test)))* y_scaler_value  

        if performance > best_performance:
            model_name = name
            best_model = model
            best_performance = r2
            best_mse = mse

        print(f'{name} Model Performance: {performance}%')
        print('\n')

    best_model_name = model_name
    res["best_performance"] = best_performance
    res["best_mse"] = best_mse

    if best_model is not None: 
        r2_training = best_model.score(X, Y)
        print(f'Best model is {best_model_name} with R² score: {best_performance} and r2_training:{r2_training}, mse {best_mse}')


    results = { "model": best_model, 
                "scaler": scaler,
                "performance": res["best_performance"], 
                "mse":res["best_mse"], 
                "y_scaler_value": y_scaler_value}

    return results


""" ---------------------------  Helper Functions --------------------"""

def preprocess(df):
  df = df[(df["512"]>44) & (df["512"]<58)] # RAED suggestion
  return df

def split_scale(df, y, inputs):
    scaler = MaxAbsScaler()
    #x = list(df.columns.difference([y]))  # List of feature columns

    # Scaling X
    train_x = scaler.fit_transform(df[inputs])
    
    # Ensure y is a DataFrame to avoid the error
    df_y = df[[y]]
    scaler_y = MaxAbsScaler()
    train_y = scaler_y.fit_transform(df_y)
    
    # Create DataFrame for train
    train = pd.DataFrame(train_x, columns=inputs)
    train[y] = train_y
    
    y_scaler_value = scaler_y.max_abs_[-1]

    train_size = int(len(train) * 0.85)
    train_df = train.iloc[:train_size]
    test_df = train.iloc[train_size:]  # for choosing the best model
    
    return train_df, test_df, scaler, y_scaler_value


def denormalize_y(y_normalized, denormalizer):
    return y_normalized * denormalizer































"""---------------------------------------------- Loading Data ----------------------------------------"""



def next_month_start(ts):
    dt = datetime.datetime.utcfromtimestamp(ts / 1000)
    if dt.month == 12:
        next_month = dt.replace(year=dt.year + 1, month=1, day=1)
    else:
        next_month = dt.replace(day=1) + datetime.timedelta(days=31)
        next_month = next_month.replace(day=1)
    return next_month.timestamp() * 1000

def load_influx_intelift(start=1680307200000, end=1693833575534, tags=[504, 512, 513], wellid=4599):
    url = 'https://15.235.80.181:443/service/Historical/getData/'
    headers = {
        "x-auth-token": "eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiIyNThkNzQzZi00NjM5LTRhNjEtODMxOC0xMTFjYzhkNzc1ZDgiLCJzdWIiOiJtb3VmaWQiLCJpZCI6NTY5LCJleHBpcmF0aW9uVGltZSI6MjAwODY5NzQ4ODE5NCwidGltZU91dCI6IjMwIiwibGFzdE5hbWUiOiJtYWF0bGFoIiwiZmlyc3ROYW1lIjoibW91ZmlkIiwiY29sbEx5bnhBY2Nlc3NDb2RlIjoiVFpIVE1XMlFJS3k1RkJHZGMzUDN3Zz09IiwiaWF0IjoxNjkzMzM3NDg4LCJleHAiOjIwMDg2OTc0ODh9.X41MKaZFVr5IvA0dSA0IbyJyX57wTpGH1zDSaIAiVm4VyIhj9PsQdKGYHqhwmW0GTR4ZSIyBnIBX71tlh64PNg",
    }

    final_data = pd.DataFrame()

    current_start = start
    while current_start < end:
        current_end = next_month_start(current_start)
        current_end = min(current_end, end)

        payload = {
            "startTime": current_start,
            "endTime": current_end,
            "tags": tags,
            "devices": [wellid]
        }

        print(f"Fetching data from {pd.to_datetime(current_start, unit='ms')} to {pd.to_datetime(current_end, unit='ms')}")
        #-> print(f"Fetching data from {current_start} to {current_end}")
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, verify=False)
        end_time = time.time()
        print(f"Request took {end_time - start_time:.2f} seconds.")

        if response.status_code == 200:
            response_data = response.json()
            print("length of the response: ", len(response_data[str(wellid)][str(tags[0])]),"\n")

            fetched_data = pd.DataFrame()

            for tag in tags:
                try:
                    tag_data = response_data[str(wellid)][str(tag)]
                    if not tag_data:
                        continue
                    timestamps = [item[0] for item in tag_data]
                    values = [item[1] for item in tag_data]
                    temp_data = pd.DataFrame({str(tag): values}, index=timestamps)
                    fetched_data = pd.concat([fetched_data, temp_data], axis=1)
                except KeyError:
                    print(f"Tag {tag} not found in the response.")
                    continue

            final_data = pd.concat([final_data, fetched_data], axis=0)

        else:
            print(f"Failed to fetch data: HTTP {response.status_code}")

        current_start = current_end + 1

    # if not final_data.empty:
    #     final_data.index = pd.to_datetime(final_data.index, unit='ms')

    return final_data
