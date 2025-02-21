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



def load_influx_data(start=1687917600000, end=1688090400000, tags= ["512","513"], wellid=4449):
        
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



def load_influx_aid(start=1694438375534, end=1693833575534, tags= [10529, 10810, 10528], wellid=4287):
        
    url = 'https://drivelynx.aidusa.com/service/Influx/ML/getData'

    headers = {
        "x-auth-token": "eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiI2NGIwZTcyYi0zOWI2LTQ0OWMtYjg3NC0zMTkyMDJhNTE5MDUiLCJzdWIiOiJtb3VmaWQiLCJpZCI6NTY5LCJleHBpcmF0aW9uVGltZSI6MTY4NTMxMDc5ODY4MSwicm9sZSI6IntcImVkaXRNb2RlXCI6dHJ1ZSxcImVkaXRcIjp0cnVlLFwicmVtb3ZlXCI6dHJ1ZSxcImVkaXRXaWRnZXRcIjp0cnVlLFwicmVtb3ZlV2lkZ2V0XCI6dHJ1ZSxcImhlYXRNYXBDb25maWdcIjp0cnVlLFwidGFic1wiOntcImNvbmZpZ3VyZVRhYnNcIjp0cnVlLFwiY3JlYXRlVGFiXCI6dHJ1ZSxcInNlbGVjdFRhYlwiOnRydWUsXCJlZGl0VGFiXCI6dHJ1ZSxcImRlbGV0ZVRhYlwiOnRydWUsXCJkZWxldGVUYWJDb25maWdcIjp0cnVlfSxcIndlbGxWaWV3XCI6e1wiY3JlYXRlV2lkZ2V0XCI6dHJ1ZSxcImVkaXRXaWRnZXRcIjp0cnVlLFwiZGVsZXRlV2lkZ2V0XCI6dHJ1ZSxcImVkaXRXaWRnZXRUYWdzXCI6dHJ1ZSxcImxpdmVSVFVBY2Nlc3NXaWRnZXRcIjp0cnVlLFwiY29ycmVsYXRpb25UYWJcIjp0cnVlLFwiZXNwUHVtcFRhYlwiOnRydWUsXCJwb3dlclRhYlwiOnRydWUsXCJhclRhYlwiOnRydWUsXCJwdW1wQ3VydmVUYWJcIjp0cnVlLFwiYW5hbHl0aWNzVGFiXCI6dHJ1ZSxcImVzcFBlcmZvcm1hbmNlVGFiXCI6dHJ1ZSxcInZpcnR1YWxGbG93TWV0ZXJUYWJcIjp0cnVlLFwid2VsbFZpZXdUYWJcIjp0cnVlLFwicmVhbFRpbWVUYWJcIjp0cnVlLFwiaGlzdG9yeVRhYlwiOnRydWUsXCJ0cmVuZEFuYWx5c2lzVGFiXCI6dHJ1ZSxcInBvbGxOb3dcIjp0cnVlLFwiYW5hbHl0aWNcIjp0cnVlLFwiYWRtaW5cIjp0cnVlfSxcImtwaVZpZXdcIjp7XCJ3ZWxsS1BJXCI6dHJ1ZSxcImZpZWxkS1BJXCI6dHJ1ZSxcImRhc2hib2FyZEtQSVwiOnRydWV9LFwiYWxsQXNzZXRzVmlld1wiOntcInNob3dJcEFkZHJlc3NcIjp0cnVlLFwic2hvd0RlY29tQXNzZXRzXCI6dHJ1ZX0sXCJyZXBvcnRWaWV3XCI6e1wiY3JlYXRlXCI6dHJ1ZSxcImVkaXRcIjp0cnVlLFwiZGVsZXRlXCI6dHJ1ZX0sXCJhbGFybVZpZXdcIjp7XCJjcmVhdGVcIjp0cnVlLFwiZWRpdFwiOnRydWUsXCJkZWxldGVcIjp0cnVlfSxcImRvY3VtZW50Vmlld1wiOntcImNyZWF0ZVwiOnRydWUsXCJlZGl0XCI6dHJ1ZSxcImRlbGV0ZVwiOnRydWV9LFwiY29udHJvbFZpZXdcIjp7XCJ2aWV3Q29udHJvbFwiOnRydWUsXCJhbGxcIjp0cnVlLFwic3RvcFwiOnRydWUsXCJzdGFydFwiOnRydWUsXCJjbGVhckZhdWx0XCI6dHJ1ZSxcInNjaGVkdWxsZVwiOnRydWUsXCJlbmFibGVDb250cm9sZVwiOnRydWUsXCJzZXRTcGVlZFwiOnRydWV9LFwiY2FtZXJhVmlld1wiOntcImxpdmVWaWRlb1wiOnRydWUsXCJjb250cm9sXCI6dHJ1ZSxcInNkQ2FyZFwiOnRydWUsXCJjb25mTW90aW9uRGV0ZWN0aW9uXCI6dHJ1ZX0sXCJhZG1pblwiOntcImdyb3VwXCI6e1wic2hvd0dyb3VwXCI6dHJ1ZSxcImNyZWF0ZVwiOnRydWUsXCJlZGl0XCI6dHJ1ZSxcImRlbGV0ZVwiOnRydWV9LFwiYXNzZXRzXCI6e1wic2hvd0Fzc2V0c1wiOnRydWUsXCJjcmVhdGVcIjp0cnVlLFwiZWRpdFwiOnRydWUsXCJlZGl0Q29tbVwiOnRydWUsXCJkZWxldGVcIjp0cnVlLFwiZGVjb21taXNzaW9uQXNzZXRzXCI6dHJ1ZSxcImx1bmNoVUVkaXRcIjp0cnVlLFwibW9kZW1BZG1pblwiOntcImNyZWF0ZVwiOnRydWUsXCJlZGl0XCI6dHJ1ZX19fX0iLCJ0aW1lT3V0IjoiMzAiLCJsYXN0TmFtZSI6Im1hYXRsYWgiLCJmaXJzdE5hbWUiOiJtb3VmaWQiLCJjb2xsTHlueEFjY2Vzc0NvZGUiOiI3QkR1OUx6S3FCOGZKdGh4Q210OHZ3PT0iLCJpYXQiOjE2ODUyNzQ3OTgsImV4cCI6MTY4NTMxMDc5OH0.ukdNn8jtcvz2WcyaP5imXiP6xFv2axQX9nvUWmUue5QXe-294B4kztm97QfpaEhw3t3UqTOFF-XMl4b1iM6I_A",
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


def load_influx_intelift(start=1694438375534, end=1693833575534, tags= [504, 512, 513], wellid=4377):
        
    url = 'https://intelift.halliburton.com/service/Influx/ML/getData'

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




def load_influx_novomet(start=1697006877000, end=1697016877000, tags= [512,504], wellid=3660):
        
    url = 'https://novomet.welllynx.com/service/Influx/ML/getData'

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



def load_influx_app(start=1699775783000, end=1699875783000, tags= ['947','808','579','963'], wellid=8064):
        
    url = 'https://app.welllynx.com/service/Influx/ML/getData'

    headers = {
        "x-auth-token":"eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiIyNThkNzQzZi00NjM5LTRhNjEtODMxOC0xMTFjYzhkNzc1ZDgiLCJzdWIiOiJtb3VmaWQiLCJpZCI6NTY5LCJleHBpcmF0aW9uVGltZSI6MjAwODY5NzQ4ODE5NCwidGltZU91dCI6IjMwIiwibGFzdE5hbWUiOiJtYWF0bGFoIiwiZmlyc3ROYW1lIjoibW91ZmlkIiwiY29sbEx5bnhBY2Nlc3NDb2RlIjoiVFpIVE1XMlFJS3k1RkJHZGMzUDN3Zz09IiwiaWF0IjoxNjkzMzM3NDg4LCJleHAiOjIwMDg2OTc0ODh9.X41MKaZFVr5IvA0dSA0IbyJyX57wTpGH1zDSaIAiVm4VyIhj9PsQdKGYHqhwmW0GTR4ZSIyBnIBX71tlh64PNg",
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



def load_influx_endurance(start=1694438375534, end=1693833575534, tags= [504, 512, 513], wellid=4377):
        
    url = 'https://endurance.welllynx.com/service/Influx/ML/getData'

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




def load_influx_landmark(start=1694438375534, end=1693833575534, tags= [504, 512, 513], wellid=4377):
        
    url = 'https://landmark.welllynx.com/service/Influx/ML/getData'

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
    if servername == "influx":
        data = load_influx_data(start= start, end= end, tags= tags, wellid= wellid)

    elif servername == "aid":
        data = load_influx_aid(start= start, end= end, tags= tags, wellid= wellid)

    elif servername == "intelift":
        data = load_influx_intelift(start= start, end= end, tags= tags, wellid= wellid)
        
    elif servername == "novomet":
        data = load_influx_novomet(start= start, end= end, tags=tags, wellid= wellid)

    elif servername == "app":
        data = load_influx_app(start= start, end= end, tags= tags, wellid= wellid)

    elif servername == "stage":
        data = load_influx_intelift(start= start, end= end, tags= tags, wellid= wellid)

    elif servername == "endurance":
        data = load_influx_endurance(start=start, end=end, tags=tags, wellid=wellid) 

    elif servername == "landmark":
        data = load_influx_landmark(start=start, end=end, tags=tags, wellid=wellid)


    return data
