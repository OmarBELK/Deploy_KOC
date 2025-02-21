import pandas as pd
import numpy as np
import random
import warnings
from scipy.interpolate import interp1d
import ast
import datetime
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.regularizers as regularizers
import os
from sources.src.upload_data import load_data
import pickle
import datetime
import warnings
warnings.filterwarnings('ignore')


## defining a function to select the columns
def preprocess(df=None, columns = None):
    
    logs = []
    status = 'pass'
    try:
        df = df[columns]
    except Exception as e:
        exception_message = str(e)
        if "[" in exception_message and "]" in exception_message:
            try:
                extracted_list = ast.literal_eval(exception_message[exception_message.index("[") : exception_message.index("]") + 1])
                print(extracted_list)
                logs.append('these tags doent exist in your selected data')
                logs.append(extracted_list)
                status='fail'
                return None, logs, status
            except (ValueError, SyntaxError) as e:
                print("The exception message does not contain a valid list")
        else:
            print("The exception message does not contain a list")

    cols = df.columns
    cols_dtypes = df.dtypes 
    is_null = df.isnull().any()
    null_cols = []
    for col in cols:
        if is_null[col]==True:
            null_cols.append(col)        
            if cols_dtypes[col]=='float':
                df[col].ffill(inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                
    df.dropna(inplace=True)

    if len(null_cols)>0:
        logs.append(f'Dataset has NULL values present at columns - {null_cols}')
        logs.append('For these columns NULL values are replaced with the last seen value of the respective column.')


    columns_to_drop = []
    for col in df.columns:
        if (df[col]==0).all():
            columns_to_drop.append(col)
            
    df = df[df.columns.difference(columns_to_drop)]
    logs.append('columns with 0 values are dropped')

    logs.append(f'data shape: {df.shape}')
    logs.append(f'data null values sum: {df.isnull().sum()}')
    return df, logs, status



""" -----------------------------------  Inference ------------------------------------ """


# def inference_random_forest(loaded_scaler=None, final_data=None,model=None,servername=None, model_id=None, wellid=4599, timestamp=None):


#     df = final_data.set_index('0')

#     data_scaledValues = loaded_scaler.transform(df.values)
#     scaled_df = pd.DataFrame(data_scaledValues,columns=df.columns)
#     scaled_df = scaled_df.set_index(df.index)
#     print("after scaling",scaled_df)

#     # make predictions
#     proba = model.predict_proba(scaled_df) 

#     stacked_array = np.array(proba[:,1])
#     # Reshape the stacked array into a 1-dimensional array
#     reshaped_array = stacked_array.reshape(-1)
#     reshaped_array = np.column_stack((np.array(final_data.iloc[:reshaped_array.shape[0],0]),reshaped_array))
#     predictions_df = pd.DataFrame(reshaped_array,columns=['Timestamp', 'Probabilities'], dtype=object)
#     predictions_df['Timestamp'] = predictions_df['Timestamp'].apply(np.int64)
#     predictions_df.to_csv(f'/mnt/{servername}/mlPrediction/{model_id}_{wellid}_{timestamp}.csv', index=False)

#     return "executed"


# def inference_auto_encoder(loaded_scaler, input_data, model, servername, model_id, wellid, timestamp):
  
#     print('------------------------- Auto-encoder inference starts ')
#     input_data = input_data.set_index('0')

#     input_ = input_data.copy() 

#     # Copy the columns and Index
#     cols = input_data.columns
#     indx = input_data.index

#     # Scale the input data
#     input_scaled = loaded_scaler.fit_transform(input_)
#     input_scaled = pd.DataFrame(input_scaled,columns=cols,index = indx)
#     reconstructions = model.predict(input_scaled)
#     mse_2 = np.array(tf.keras.losses.mae(reconstructions, input_scaled))
#     input_data['Probabilities'] = predict_probs(mse_2,95)
#     input_data['Probabilities'] = input_data['Probabilities'].clip(lower=0, upper=100)

#     output_df = input_data


#     output_df = output_df.reset_index().rename(columns={'0': 'Timestamp'})[['Timestamp', 'Probabilities']]

#     path_to_save_in = f'/mnt/{servername}/mlPrediction/{model_id}_{wellid}_{timestamp}.csv'
#     output_df.to_csv(path_to_save_in, index=False)
#     print(f'------------------------- File is saved  succeffully at {path_to_save_in} \n')

#     return "executed"
    



# def inference_isolation_forest(final_data=None, model=None, servername=None, model_id=None, wellid=4599, timestamp=None):

#     df = final_data.set_index('0')
#     scores_proba = model.decision_function(df)
#     proba = (scores_proba.max() - scores_proba) / (scores_proba.max() - scores_proba.min()) * 100
#     print("------------------------- proba is calculated successfully.", proba[:5]," ...")
#     stacked_array = np.array(proba)
#     # Reshape the stacked array into a 1-dimensional array
#     reshaped_array = stacked_array.reshape(-1)
#     reshaped_array = np.column_stack((np.array(final_data.iloc[:reshaped_array.shape[0],0]),reshaped_array))
#     predictions_df = pd.DataFrame(reshaped_array,columns=['Timestamp', 'Probabilities'], dtype=object)
#     predictions_df['Timestamp'] = predictions_df['Timestamp'].apply(np.int64)
#     path_to_save_in = f'/mnt/{servername}/mlPrediction/{model_id}_{wellid}_{timestamp}.csv'
#     predictions_df.to_csv(path_to_save_in, index=False)
#     print(f'------------------------- File is saved  succeffully at {path_to_save_in} \n')

#     return "executed"




import paho.mqtt.client as mqtt
import json
from app.config.mqtt_config import CLIENT_CONFIG



def publish_to_mqtt(config, content):
    """
    Publishes a message to an MQTT broker.

    Parameters:
        config (dict): A dictionary containing MQTT configuration.
            - broker (str): The MQTT broker's address.
            - port (int): The port number to connect to the broker.
            - topic (str): The MQTT topic to publish to.
            - username (str): Username for authentication.
            - password (str): Password for authentication.
        content (str): The message content to publish.

    Returns:
        str: Status message indicating success or error.
    """
    broker = config["broker"]
    port = config["port"]
    topic = config["topic"]
    username = config["username"]
    password = config["password"] 


    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        # Connect to MQTT broker
        client.connect(broker, port, 60)
        
        # Start the loop to process network traffic and dispatch callbacks
        client.loop_start()

        # Publish the message to the MQTT topic
        result = client.publish(topic, content)

        # Wait for a moment to ensure the message is sent
        result.wait_for_publish()

        if result.is_published():
            print(f"Data sent successfully to MQTT topic: {topic}")
            return "Data sent successfully."
        else:
            print(f"Data not sent to MQTT topic: {topic}")
            return "Failed to send data."
        
    except Exception as e:
        error_message = f"Error sending data to MQTT: {str(e)}"
        print(error_message)
        return error_message
        
    finally:
        # Stop the loop and disconnect from the broker
        client.loop_stop()
        client.disconnect()





def inference_auto_encoder(tag_id, loaded_scaler, input_data, model, servername, model_id, wellid, timestamp):

    
    print('------------------------- Auto-encoder inference starts ')
    input_data = input_data.set_index('0')

    input_ = input_data.copy() 

    # Copy the columns and Index
    cols = input_data.columns
    indx = input_data.index

    # Scale the input data
    input_scaled = loaded_scaler.fit_transform(input_)
    input_scaled = pd.DataFrame(input_scaled,columns=cols,index = indx)
    reconstructions = model.predict(input_scaled)
    mse_2 = np.array(tf.keras.losses.mae(reconstructions, input_scaled))
    input_data['Probabilities'] = predict_probs(mse_2,95)
    input_data['Probabilities'] = input_data['Probabilities'].clip(lower=0, upper=100)

    # Format the data
    output_df = input_data.reset_index()[['0', 'Probabilities']]
    output_df.columns = ['0', str(tag_id)]  # Rename columns to match format

    # Create the content string
    content = f"{wellid}_datalog_{servername}_{model_id}.csv\n"  # Header line
    content += f"0,{str(tag_id)}\n"  # Column names
    # Add data rows
    content += output_df.to_csv(index=False, header=False)

    # Load the Client Config
    config = client_config[str(servername)]

    # Publish to MQTT
    mqtt_status = publish_to_mqtt(config, content)

    return mqtt_status



def inference_random_forest(tag_id,loaded_scaler=None, final_data=None, model=None, servername=None, model_id=None, wellid=None, timestamp=None):

    df = final_data.set_index('0')

    data_scaledValues = loaded_scaler.transform(df.values)
    scaled_df = pd.DataFrame(data_scaledValues, columns=df.columns)
    scaled_df = scaled_df.set_index(df.index)
    print("after scaling", scaled_df)

    # make predictions
    proba = model.predict_proba(scaled_df) 

    stacked_array = np.array(proba[:,1])
    reshaped_array = stacked_array.reshape(-1)
    reshaped_array = np.column_stack((np.array(final_data.iloc[:reshaped_array.shape[0],0]), reshaped_array))
    
    # Create DataFrame with the correct format
    predictions_df = pd.DataFrame(reshaped_array, columns=['0', str(tag_id)])
    predictions_df['0'] = predictions_df['0'].apply(np.int64)
    predictions_df[(str(tag_id))] = predictions_df[str(tag_id)] * 100  # Convert to percentage if needed

    # Create the content string
    content = f"{wellid}_datalog_{servername}_{model_id}.csv\n"  # Header line
    content += f"0,{str(tag_id)}\n"  # Column names
    # Add data rows
    content += predictions_df.to_csv(index=False, header=False)

    # client Configuration
    config = client_config[str(servername)]

    # Publish to MQTT
    mqtt_status = publish_to_mqtt(config, content)

    return mqtt_status


def inference_isolation_forest(final_data=None, model=None, servername=None, model_id=None, wellid=4599, timestamp=None):

    df = final_data.set_index('0')
    scores_proba = model.decision_function(df)
    proba = (scores_proba.max() - scores_proba) / (scores_proba.max() - scores_proba.min()) * 100
    print("------------------------- proba is calculated successfully.", proba[:5]," ...")
    stacked_array = np.array(proba)
    # Reshape the stacked array into a 1-dimensional array
    reshaped_array = stacked_array.reshape(-1)
    reshaped_array = np.column_stack((np.array(final_data.iloc[:reshaped_array.shape[0],0]),reshaped_array))
    predictions_df = pd.DataFrame(reshaped_array,columns=['Timestamp', 'Probabilities'], dtype=object)
    predictions_df['Timestamp'] = predictions_df['Timestamp'].apply(np.int64)
    path_to_save_in = f'/mnt/{servername}/mlPrediction/{model_id}_{wellid}_{timestamp}.csv'
    predictions_df.to_csv(path_to_save_in, index=False)
    print(f'------------------------- File is saved  succeffully at {path_to_save_in} \n')

    return "executed"


"""---------------------------------------      Backtesting       ------------------------------------"""


min_max_scaler = preprocessing.MinMaxScaler()

def backtesting_auto_encoder(loaded_scaler= None,df = None,columns = None, threshold=None, model=None):

    # Set the timestamp column as the index
    df = df.set_index('0')
    df = df.astype('float')
    ##---- Scale the values
    try:
        data_scaledValues = loaded_scaler.transform(df.values)
        scaled_df = pd.DataFrame(data_scaledValues,columns=df.columns)
        scaled_df = scaled_df.set_index(df.index)
        print("df after scaling", scaled_df)
        print("-------------------------The scaling runs successfully")
    except Exception as e:
        print(e)
        print("Error in scaling the data")

    # predicting the probabilities with the pre defined threshold 
    try:
        reconstructions = model.predict(df)
        mse_2 = np.array(tf.keras.losses.mae(reconstructions, df))
        df['probability'] = predict_probs(mse_2,95)
    except Exception as e:
        print(e)

    return df

def backtesting_random_forest(df, columns = None,model=None, loaded_scaler=None):
    df = df.set_index('0')
    data_scaledValues = loaded_scaler.transform(df.values)
    scaled_df = pd.DataFrame(data_scaledValues,columns=df.columns)
    scaled_df = scaled_df.set_index(df.index)
    proba = model.predict_proba(scaled_df) 
    df['probability'] = proba[:, 1] * 100
    df_final = df
    return df_final

def backtesting_svc(df, columns = None,model=None, loaded_scaler=None):
    df = df.set_index('0')
    data_scaledValues = loaded_scaler.transform(df.values)
    scaled_df = pd.DataFrame(data_scaledValues,columns=df.columns)
    scaled_df = scaled_df.set_index(df.index)
    proba = model.predict_proba(scaled_df) 
    df['probability'] = proba[:, 1] * 100
    df_final = df
    return df_final


def predict_probs(losses, threshold):
    probabilities = []
    for loss in losses:
        if loss > threshold:
            probabilities.append(loss*100)
        else:
            probabilities.append((1 - loss)*100)
    return np.array(probabilities)



def perform_backtest(**kwargs):
    print("---------------------- backtesting started")

    algo_name = kwargs["algo_name"]
    wellid = kwargs["wellid"]
    servername = kwargs["servername"]
    inputs = kwargs["columns"]
    trained_at = kwargs["trained_at_ms"]
    user_id = kwargs['user_id']
    start = kwargs["start"]
    end = kwargs["end"]


    if algo_name=="auto-encoder":
        # Update the filename of the trained model

        unique_filename = f"model_{servername}_{trained_at}.h5"
        model_file_path = os.path.join(".", unique_filename)
        print("model file path ---- ", model_file_path)


        scaler_filename = f"scaler_{servername}_{trained_at}.pkl"
        scaler_file_path = os.path.join(".", scaler_filename)

        # Load the saved scaler from the file
        with open(scaler_file_path, 'rb') as f:
            loaded_scaler = pickle.load(f)
            print("------------------------- The scaler file loaded successfully.")


        # Check if the file exists
        if os.path.exists(model_file_path):
            print("------------------------- The model file exists ")
        else:
            print(" The file does not exist. \n")

        # load the model
        try:
            loaded_model = tf.keras.models.load_model(model_file_path)
            print("--------------------- The model loaded successfully ")
        except Exception as e:
            print("Error in Loading model")
        
        # load the requested Historical data
        try:
            df = load_data(servername=servername, start=start, end=end, tags=inputs, wellid=wellid)
            print('------------------------- Historical Data is loaded successfully')
        except Exception as  e:
            print(e)
            print("Error in Loading Historical Data")
        
        # process the df data to be test
        test = df.set_index('0')
        test_ = test.copy()

        try:
            cols = test.columns
            indx = test.index
            test = loaded_scaler.transform(test)
            test = pd.DataFrame(test,columns=cols,index = indx)
            reconstructions = loaded_model.predict(test)
            mse_2 = np.array(tf.keras.losses.mae(reconstructions, test))
            test_['probability'] = predict_probs(mse_2,95)
            test_['probability'] = test_['probability'].clip(lower=0, upper=100)
            df_results = test_
            print("results of backtesting are \n", df_results)
        except Exception as e:
            print("\n Error in Backtesting (predicting) process")
            print(e)
            
            

    elif algo_name=="random-forest":

        # Update the filename of the trained model
        unique_filename = f"model_{servername}_{trained_at}.pkl"
        model_file_path = os.path.join(".", unique_filename)

        # Check if the file exists
        if os.path.exists(model_file_path):
            print("------------------------- The model file exists.")
        else:
            print("------------------------- The file does not exist.")

        scaler_filename = f"scaler_{servername}_{trained_at}.pkl"
        scaler_file_path = os.path.join(".", scaler_filename)

        # Load the saved scaler from the file
        with open(scaler_file_path, 'rb') as f:
            loaded_scaler = pickle.load(f)
            print("------------------------- The scaler file loaded successfully.")

        # load the model
        try:
            # Load the model from file
            with open(model_file_path, 'rb') as file:
                loaded_model = pickle.load(file)
                print("------------------------- The model file loaded successfully.")
        except Exception as e:
            print("Error in Loading model")
        
        # search  for the requested Historical data
        try:
            df = load_data(servername=servername,start=start, end=end, tags=inputs, wellid=wellid)
            print('------------------------- Historical Data is loaded successfully')
        except Exception as  e:
            print(e)
            print("Error in Loading Historical Data")
    
        try:
            # backtesting function
            df_results  = backtesting_random_forest(df = df, columns = inputs,  model= loaded_model,
                                                    loaded_scaler=loaded_scaler)
            print('------------------------- df results is returned successfully')
        except Exception as e:
            print(e)
            print("Data are not enough to be processed")
        
    elif algo_name=="svc":

        # Update the filename of the trained model
        unique_filename = f"{wellid}_{servername}_{trained_at}.pkl"
        model_file_path = os.path.join(".", unique_filename)

        # Check if the file exists
        if os.path.exists(model_file_path):
            print("------------------------- The model file exists.")
        else:
            print("------------------------- The file does not exist.")

        scaler_filename = f"scaler_{wellid}_{servername}_{trained_at}.pkl"
        scaler_file_path = os.path.join(".", scaler_filename)

        # Load the saved scaler from the file
        with open(scaler_file_path, 'rb') as f:
            loaded_scaler = pickle.load(f)
            print("------------------------- The scaler file loaded successfully.")

        # load the model
        try:
            # Load the model from file
            with open(model_file_path, 'rb') as file:
                loaded_model = pickle.load(file)
                print("------------------------- The model file loaded successfully.")
        except Exception as e:
            print("Error in Loading model")
        
        # search  for the requested Historical data
        try:
            df = load_data(servername=servername,start=start, end=end, tags=inputs, wellid=wellid)
            print('------------------------- Historical Data is loaded successfully')
        except Exception as  e:
            print(e)
            print("Error in Loading Historical Data")
    
        try:
            # backtesting function
            df_results  = backtesting_svc(df = df, columns = inputs,  model= loaded_model,
                                                    loaded_scaler=loaded_scaler)
            print('------------------------- df results is returned successfully')
        except Exception as e:
            print(e)
            print("Data are not enough to be processed")


    if algo_name == "isolation-forest":

        # Update the filename of the trained model
        unique_filename = f"{wellid}_{servername}_{trained_at}.pkl"
        model_file_path = os.path.join(".", unique_filename)

        # load the model
        try:
            # Load the model from file
            with open(model_file_path, 'rb') as file:
                loaded_model = pickle.load(file)
                print("------------------------- The model file loaded successfully.")
        except Exception as e:
            print("Error in Loading model")
        
        # search  for the requested Historical data
        try:
            df = load_data(servername=servername,start=start, end=end, tags=inputs, wellid=wellid)
            print('------------------------- Historical Data is loaded successfully')
        except Exception as  e:
            print(e)
            print("Error in Loading Historical Data")
    
        try:
            # backtesting function
            test = df.set_index('0')
            scores_proba = loaded_model.decision_function(test)
            test['probability'] = (scores_proba.max() - scores_proba) / (scores_proba.max() - scores_proba.min()) * 100
            df_results = test
            print('------------------------- df results is returned successfully')
        except Exception as e:
            print(e)
            print("Error in backtesting part")



    #converting the df_results requested by abdel
    df = pd.DataFrame(df_results)

    return df



"""------------------------------------ Perform Backtestig for Admins ------------------------"""


def perform_backtest_admin(**kwargs):
    print("---------------------- backtesting started")
    model_id = kwargs["model_id"]
    algo_name = kwargs["algo_name"]
    wellid = kwargs["wellid"]
    servername = kwargs["servername"]
    inputs = kwargs["columns"]
    start = kwargs["start"]
    end = kwargs["end"]


    if algo_name=="auto-encoder":
        
        # path the filename of the trained model
        unique_filename = f"/mnt/models/model_{model_id}.h5"
        model_file_path = os.path.join(".", unique_filename)
        print("model file path ---- ", model_file_path)


        # Check if the file exists
        if os.path.exists(model_file_path):
            print("------------------------- The model file exists ")
        else:
            print(" The file does not exist. \n")

        scaler_filename = f"/mnt/scalers/scaler_{model_id}.pkl"
        scaler_file_path = os.path.join(".", scaler_filename)

        # Load the saved scaler from the file
        with open(scaler_file_path, 'rb') as f:
            loaded_scaler = pickle.load(f)
            print("------------------------- The scaler file loaded successfully.")

        # load the model
        try:
            loaded_model = tf.keras.models.load_model(model_file_path)
            print("--------------------- The model loaded successfully ")
        except Exception as e:
            print("Error in Loading model")
        
        print("inputs", inputs)
        # load the requested Historical data
        try:
            df = load_data(servername=servername, start=start, end=end, tags=inputs, wellid=wellid)
            print('------------------------- Historical Data is loaded successfully')
        except Exception as  e:
            print("Exception is: ",e)
            print("Error in Loading Historical Data")
        
        # process the df data to be test
        test = df.set_index('0')
        test_ = test.copy()

        try:
            cols = test.columns
            indx = test.index
            test = loaded_scaler.transform(test)
            test = pd.DataFrame(test,columns=cols,index = indx)
            reconstructions = loaded_model.predict(test)
            mse_2 = np.array(tf.keras.losses.mae(reconstructions, test))
            test_['probability'] = predict_probs(mse_2,95)
            df_results = test_
            print("results of backtesting are \n", df_results)
        except Exception as e:
            print("\n Error in Backtesting (predicting) process")
            print(e)
    

    #converting the df_results requested by abdel
    df = pd.DataFrame(df_results)

    return df

"""------------------------------------ End of Backtesting for Admins ------------------------"""

from core.models import TestingStatus     


def backtesting_isolation_forest(data= None, loaded_model= None, testing_status= None):
    test = data.set_index('0')

    window_size = 10000  # Number of rows per window

    if len(test) < window_size:
        # If the length of test_ is less than window_size, process it in a single window
        total_windows = 1
    else:
        total_windows = len(test) // window_size

    # Check if there are remaining data points that don't fit into a full window
    if len(test) % window_size != 0:
        total_windows += 1
        print("len(test) % window size ", len(test) % window_size)

    print("total windows: ", total_windows)

    all_probs = []

    for window_index in range(total_windows):
        start_idx = window_index * window_size
        end_idx = min((window_index + 1) * window_size, len(test))

        window_data = test.iloc[start_idx:end_idx]
        print("window data shape ",window_data.shape)
        if window_data.shape[0]==0:
            break
        # Perform your processing on the window_data here
        scores_proba = loaded_model.decision_function(window_data)
        probability_window = (scores_proba.max() - scores_proba) / (scores_proba.max() - scores_proba.min()) * 100
        print("probability window: ", probability_window)
        all_probs.append(probability_window)

        # Calculate the progress dynamically
        progress = (window_index + 1) / total_windows * 100
        print("progress: ",progress)

        # Update progress here (e.g., print or log progress)
        print(f"Processed Window {window_index + 1}/{total_windows}, Progress: {progress:.2f}%")

        # Update the progress attribute in the TestingStatus model
        testing_status.progress = progress
        testing_status.save()

    combined_probs = np.concatenate(all_probs, axis=0)
    print('combined probs: ',combined_probs)

    test["Probabilities"] = combined_probs

    return test




""" ------------------------------------------ Artifacts -----------------------------------------"""


def preprocess_new(df=None, columns = None):
    
    logs = []
    status = 'pass'


    # calculate percentage of missing values for each column
    missing_percentages = df.isna().mean() * 100

    # filter out columns with less than 10% missing values
    threshold = 40

    # Selected Columns
    all_columns = missing_percentages[missing_percentages < threshold].index

    selected_columns = [tag for tag in all_columns if tag in columns]

    selected_columns.insert(0, "0")

    df = df[selected_columns]

    # check if each row has only valid values
    only_values = df.notnull().all(axis=1)

    df = df[only_values]

    # Divide Unix timestamp by 1000 to convert it to seconds
    df['0'] = df['0'] / 1000

    # Convert timestamp to datetime object
    df['0'] = df['0'].apply(lambda x: datetime.datetime.fromtimestamp(x))


    # Set the timestamp column as the index
    df = df.set_index('0')

    # Resample the data into 1-minute intervals and take the mean of the values
    df_resampled = df.resample('1T').mean().interpolate(method='linear')

    df = df_resampled

    print(df.iloc[:5,:5])

    return df, logs, status