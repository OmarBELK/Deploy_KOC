import numpy as np
import pandas as pd 
from datetime import datetime
from sklearn import preprocessing
import requests



##### function for search_logic

def search_logic(servername="stage", wellid = None, start=None, end=None, tags=None):

    df = pd.DataFrame()
    # Convert timestamp to year
    start_year = datetime.utcfromtimestamp(start/1000).strftime('%Y')
    # Convert timestamp to month
    start_month = datetime.utcfromtimestamp(start/1000).strftime('%m')
    # Convert timestamp to year
    end_year = datetime.utcfromtimestamp(end/1000).strftime('%Y')
    # Convert timestamp to month
    end_month = datetime.utcfromtimestamp(end/1000).strftime('%m')
    # Load start-time file from direction
    if(int(start_year)<int(end_year)):
        print('first condition')
        for i in range(int(start_month),13):
            directions = str("/app/sources/mnt/"+ servername +"/"+ str(wellid) + "/history/" + str(start_year) + "/" + str(i).zfill(2)  + ".csv")
            print(directions)
            var = pd.read_csv(directions)
            df = df.append(var)
        for i in range(1,int(end_month)+1):
            directions = str("/app/sources/mnt/"+ str(servername) +"/"+ str(wellid) + "/history/" + str(end_year) + "/" + str(i).zfill(2)  + ".csv")
            print(directions)
            var = pd.read_csv(directions)
            df = df.append(var)
            
    elif( int(start_year) == int(end_year)):
        print('-------second condition: both of time stamps are in the same year')
        for i in range(int(start_month),int(end_month)+1):
            directions = "/app/sources/mnt/"+ servername +"/"+ str(wellid) + "/history/" + str(start_year) + "/" + str(i).zfill(2) + ".csv"
            print("loaded directory is :",directions)
            data = pd.read_csv(directions)
            data = data[tags]            
            data = data.dropna()
            df = df.append(data)
    else:
        print('error')

    ##### search for [start,end] in df

    data_final = df[( df['0'] >= start ) & ( df['0'] <= end ) ]
    data_final = data_final.reset_index(drop=True)

    return data_final




def load_influx_koc(start=1694438375534, end=1693833575534, tags= [504, 512, 513], wellid=4377):
        
    url = 'https://stage.welllynx.com/service/Influx/ML/getData'

    headers = {
        "x-auth-token": "eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiIyNThkNzQzZi00NjM5LTRhNjEtODMxOC0xMTFjYzhkNzc1ZDgiLCJzdWIiOiJtb3VmaWQiLCJpZCI6NTY5LCJleHBpcmF0aW9uVGltZSI6MjAwODY5NzQ4ODE5NCwidGltZU91dCI6IjMwIiwibGFzdE5hbWUiOiJtYWF0bGFoIiwiZmlyc3ROYW1lIjoibW91ZmlkIiwiY29sbEx5bnhBY2Nlc3NDb2RlIjoiVFpIVE1XMlFJS3k1RkJHZGMzUDN3Zz09IiwiaWF0IjoxNjkzMzM3NDg4LCJleHAiOjIwMDg2OTc0ODh9.X41MKaZFVr5IvA0dSA0IbyJyX57wTpGH1zDSaIAiVm4VyIhj9PsQdKGYHqhwmW0GTR4ZSIyBnIBX71tlh64PNg",
    }

    payload = {
            "startTime": start,
            "endTime": end,
            "tags": tags,
            "devices": [wellid]
            }

    response = requests.post(url, headers=headers, json=payload, verify=False)

    # Access the response data
    response_data = response.json()

    j = 0
    for i in tags:
        if j == 0:
            data = pd.Series(response_data[str(wellid)][str(i)])
            data = pd.DataFrame(data.apply(lambda x: [str(val) for val in x]).to_list())
            data = data.rename(columns={1: str(i)})
        else:
            data[str(i)] = pd.DataFrame(response_data[str(wellid)][str(i)])[1]  
        j=1

    data = data.rename(columns={0: '0'})
    # convert date 
    data['0'] = data['0'].astype(float)
    data['0'] = data['0'].apply('{:.0f}'.format)

    data.dropna(inplace=True)
        
    return data



def load_data(servername = None, start= 1687917600000, end= 1688090400000, tags= ["512","513"], wellid=4449):
    if servername == "koc":
        data = load_influx_koc(start= start, end= end, tags= tags, wellid= wellid)

    return data
