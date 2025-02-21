
import requests
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing


def load_influx_data(start=1687917600000, end=1688090400000, tags= ["512","513"], wellid=4616):

    url = 'http://15.235.80.181:6655/service/Influx/ML/getData'

    headers = {
        "x-auth-token": "eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiI2NGIwZTcyYi0zOWI2LTQ0OWMtYjg3NC0zMTkyMDJhNTE5MDUiLCJzdWIiOiJtb3VmaWQiLCJpZCI6NTY5LCJleHBpcmF0aW9uVGltZSI6MTY4NTMxMDc5ODY4MSwicm9sZSI6IntcImVkaXRNb2RlXCI6dHJ1ZSxcImVkaXRcIjp0cnVlLFwicmVtb3ZlXCI6dHJ1ZSxcImVkaXRXaWRnZXRcIjp0cnVlLFwicmVtb3ZlV2lkZ2V0XCI6dHJ1ZSxcImhlYXRNYXBDb25maWdcIjp0cnVlLFwidGFic1wiOntcImNvbmZpZ3VyZVRhYnNcIjp0cnVlLFwiY3JlYXRlVGFiXCI6dHJ1ZSxcInNlbGVjdFRhYlwiOnRydWUsXCJlZGl0VGFiXCI6dHJ1ZSxcImRlbGV0ZVRhYlwiOnRydWUsXCJkZWxldGVUYWJDb25maWdcIjp0cnVlfSxcIndlbGxWaWV3XCI6e1wiY3JlYXRlV2lkZ2V0XCI6dHJ1ZSxcImVkaXRXaWRnZXRcIjp0cnVlLFwiZGVsZXRlV2lkZ2V0XCI6dHJ1ZSxcImVkaXRXaWRnZXRUYWdzXCI6dHJ1ZSxcImxpdmVSVFVBY2Nlc3NXaWRnZXRcIjp0cnVlLFwiY29ycmVsYXRpb25UYWJcIjp0cnVlLFwiZXNwUHVtcFRhYlwiOnRydWUsXCJwb3dlclRhYlwiOnRydWUsXCJhclRhYlwiOnRydWUsXCJwdW1wQ3VydmVUYWJcIjp0cnVlLFwiYW5hbHl0aWNzVGFiXCI6dHJ1ZSxcImVzcFBlcmZvcm1hbmNlVGFiXCI6dHJ1ZSxcInZpcnR1YWxGbG93TWV0ZXJUYWJcIjp0cnVlLFwid2VsbFZpZXdUYWJcIjp0cnVlLFwicmVhbFRpbWVUYWJcIjp0cnVlLFwiaGlzdG9yeVRhYlwiOnRydWUsXCJ0cmVuZEFuYWx5c2lzVGFiXCI6dHJ1ZSxcInBvbGxOb3dcIjp0cnVlLFwiYW5hbHl0aWNcIjp0cnVlLFwiYWRtaW5cIjp0cnVlfSxcImtwaVZpZXdcIjp7XCJ3ZWxsS1BJXCI6dHJ1ZSxcImZpZWxkS1BJXCI6dHJ1ZSxcImRhc2hib2FyZEtQSVwiOnRydWV9LFwiYWxsQXNzZXRzVmlld1wiOntcInNob3dJcEFkZHJlc3NcIjp0cnVlLFwic2hvd0RlY29tQXNzZXRzXCI6dHJ1ZX0sXCJyZXBvcnRWaWV3XCI6e1wiY3JlYXRlXCI6dHJ1ZSxcImVkaXRcIjp0cnVlLFwiZGVsZXRlXCI6dHJ1ZX0sXCJhbGFybVZpZXdcIjp7XCJjcmVhdGVcIjp0cnVlLFwiZWRpdFwiOnRydWUsXCJkZWxldGVcIjp0cnVlfSxcImRvY3VtZW50Vmlld1wiOntcImNyZWF0ZVwiOnRydWUsXCJlZGl0XCI6dHJ1ZSxcImRlbGV0ZVwiOnRydWV9LFwiY29udHJvbFZpZXdcIjp7XCJ2aWV3Q29udHJvbFwiOnRydWUsXCJhbGxcIjp0cnVlLFwic3RvcFwiOnRydWUsXCJzdGFydFwiOnRydWUsXCJjbGVhckZhdWx0XCI6dHJ1ZSxcInNjaGVkdWxsZVwiOnRydWUsXCJlbmFibGVDb250cm9sZVwiOnRydWUsXCJzZXRTcGVlZFwiOnRydWV9LFwiY2FtZXJhVmlld1wiOntcImxpdmVWaWRlb1wiOnRydWUsXCJjb250cm9sXCI6dHJ1ZSxcInNkQ2FyZFwiOnRydWUsXCJjb25mTW90aW9uRGV0ZWN0aW9uXCI6dHJ1ZX0sXCJhZG1pblwiOntcImdyb3VwXCI6e1wic2hvd0dyb3VwXCI6dHJ1ZSxcImNyZWF0ZVwiOnRydWUsXCJlZGl0XCI6dHJ1ZSxcImRlbGV0ZVwiOnRydWV9LFwiYXNzZXRzXCI6e1wic2hvd0Fzc2V0c1wiOnRydWUsXCJjcmVhdGVcIjp0cnVlLFwiZWRpdFwiOnRydWUsXCJlZGl0Q29tbVwiOnRydWUsXCJkZWxldGVcIjp0cnVlLFwiZGVjb21taXNzaW9uQXNzZXRzXCI6dHJ1ZSxcImx1bmNoVUVkaXRcIjp0cnVlLFwibW9kZW1BZG1pblwiOntcImNyZWF0ZVwiOnRydWUsXCJlZGl0XCI6dHJ1ZX19fX0iLCJ0aW1lT3V0IjoiMzAiLCJsYXN0TmFtZSI6Im1hYXRsYWgiLCJmaXJzdE5hbWUiOiJtb3VmaWQiLCJjb2xsTHlueEFjY2Vzc0NvZGUiOiI3QkR1OUx6S3FCOGZKdGh4Q210OHZ3PT0iLCJpYXQiOjE2ODUyNzQ3OTgsImV4cCI6MTY4NTMxMDc5OH0.ukdNn8jtcvz2WcyaP5imXiP6xFv2axQX9nvUWmUue5QXe-294B4kztm97QfpaEhw3t3UqTOFF-XMl4b1iM6I_A",
    }


    payload = {
            "startTime": start,
            "endTime": end,
            "tags": tags,
            "devices": [wellid]
            }
    
    response = requests.post(url, headers=headers, json=payload)

    # Access the response data
    response_data = response.json()

    # Extract the data for the desired columns
    df_data = {}

    for outer_key, inner_dict in response_data.items():
        for inner_key, inner_list in inner_dict.items():
            column_name = f"{inner_key}"
            column_values = [data[1] for data in inner_list]
            df_data.setdefault(column_name, []).extend(column_values)
        
        timestamp_values = [data[0] for data in  inner_list]

        df_data['0'] = timestamp_values

    # Step 1: Determine the maximum length
    max_length = max(len(values) for values in df_data.values())

    # Step 2: Extend values with NaN to match the maximum length
    for key in df_data:
        df_data[key] += [float('nan')] * (max_length - len(df_data[key]))

    # Step 3: Convert the modified dictionary to a DataFrame
    data = pd.DataFrame(df_data)
    data.dropna(inplace=True)

    return data


def manual_pipeline(wellid=4616,start=1682917200000, end=1685595599000):
    
    # Select the desired columns
    tag_list = ['0','512', '989', '2317', '2318']
    
    #start=1682917200000 
    #end=1685595599000

    data=load_influx_data(wellid=wellid,tags=tag_list, start=start, end=end)

    #  Frequency , well head pressure, injection well pressure, injection temp
    data = data[tag_list]

    # Drop rows with missing values
    data = data.dropna()

    data = data.set_index('0')

    return data




def scaling(data):
    # Scaling data    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data.values)
    data_scaledValues = min_max_scaler.transform(data.values)
    scaled_df = pd.DataFrame(data_scaledValues,columns=data.columns)
    scaled_df = scaled_df.set_index(data.index)
    
    return scaled_df, min_max_scaler
