from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework import status as st
from django.http import JsonResponse
from rest_framework.decorators import api_view
from .models import  TrainingMetrics, TrainingStatus, TrainedModel
from sources.src.upload_data import search_logic
from .training import training
from sources.src.preprocess_data import inference_auto_encoder, preprocess,inference_random_forest, inference_isolation_forest
from sources.src.time_noise import generate_synthetic_data
from datetime import datetime
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import pandas as pd
import threading
import psutil
import json
import os
import joblib
import shutil
import pickle
import numpy as np
import ast
from sklearn import preprocessing
from tensorflow.keras.models import save_model
from sources.src.upload_data import load_data
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import threading
from sources.src.preprocess_data import perform_backtest, perform_backtest_admin
from django.core.cache import cache
import tempfile
from .models import  TestingStatus
from django.utils import timezone
from sources.src.upload_data import load_data
from .training import run_autoencoder_production




@api_view(["POST"])
def run_manual_autoencoder(request):

    if request.method=="POST":
        data = json.loads(request.body)

        # Gathering parameters
        timestamp = int(data["timestamp"])
        model_period = str(data["model_period"])
        wellid = data["wellid"]
        servername = str(data['servername'])
        start = timestamp-(3600000*6)
        end = timestamp

        # Load Data
        try:
            tags = ["507", "504", "508", "505", "511"]
            df = load_data(servername=servername, start=start, end=end, tags= tags, wellid=wellid)
            print(f'------------------------- Data is loaded  succeffully and has a head of{df.head()} \n')
        except Exception as e:
            print(e)
            return JsonResponse("Error in Loading Data", status=400, safe=False)

        # Call the AE Method to run the Models
        output_df = run_autoencoder_production(input_data=df, model_period=model_period)
        print(f'------------------------- Model Run succeffully\n {output_df}')

        output_df = output_df.reset_index().rename(columns={'0': 'Timestamp'})[['Timestamp', 'Probabilities']]

        path_to_save_in = f'/mnt/{servername}/mlPrediction/ESP_Normal_{model_period}_{wellid}_{timestamp}.csv'
        output_df.to_csv(path_to_save_in, index=False)
        print(f'------------------------- File is saved  succeffully at {path_to_save_in} \n')

        # Return a response to the client
        response_data = {'message': 'Model run successfully'}
        return HttpResponse(json.dumps(response_data), content_type='application/json')
    
    else:
        response_data = {'error': 'Invalid request method'}
        return HttpResponse(json.dumps(response_data), content_type='application/json', status=400)


running_tasks = {}
cancel_events = {}

@api_view(["POST"])
def start_training(request):
    # Parse the JSON data in the request body to a dictionary
    data = json.loads(request.body)

    algo_name = data["algo_name"]

    # Create a new event object for this training task
    cancel_event = threading.Event()

    if algo_name == "auto-encoder":
        # Extract the parameters for the auto-encoder 
        kwargs = {
            'algo_name':data['algo_name'],
            'servername': data['servername'],
            'pattern_ranges' : data['pattern_ranges'],
            'tags': data['tags'],
            'batch_size': data['batch_size'],
            'encoding_dim': data['encoding_dim'],
            'epochs': data['epochs'],
            'model_name': data['model_name'],
            'model_description': data['model_description'],
            'user_id': data['user_id'],
            'model_type': data['model_type'],
            'is_generic': data['is_generic'],
        }

    elif algo_name == "random-forest":
        # Extract the parameters for the random forest
        kwargs = {
            'algo_name':data['algo_name'],
            'servername': data['servername'],
            'pattern_ranges' : data['pattern_ranges'],
            'normal_ranges' : data['normal_ranges'],
            'tags': data['tags'],
            'model_name': data['model_name'],
            'model_description': data['model_description'],
            'user_id': data['user_id'],
            'model_type': data['model_type'],
            'is_generic': data['is_generic'],
        }

    elif algo_name=="isolation-forest":
        # Extract the parameters for the isolation forest
        kwargs = {
            'algo_name':data['algo_name'],
            'wellid': data['wellid'],
            'servername': data['servername'],
            'start': data['start'],
            'end': data['end'],
            'tags': data['tags'],
            'samples': data['samples'],
            'noise': data['noise'],
            'n_trees': data['n_trees'],
            'max_samples': data['max_samples'],
            'contamination': data['contamination'],
            'model_name': data['model_name'],
            'model_description': data['model_description'],
            'user_id': data['user_id'],
            'model_type': data['model_type'],
            'is_generic': data['is_generic'],
        }
    elif algo_name == "svc":
        # Extract the parameters for the support vector machine
        kwargs = {
            'algo_name':data['algo_name'],
            'wellid': data['wellid'],
            'servername': data['servername'],
            'pattern_ranges' : data['pattern_ranges'],
            'normal_ranges' : data['normal_ranges'],
            'tags': data['tags'],
            'model_name': data['model_name'],
            'model_description': data['model_description'],
            'user_id': data['user_id'],
            'model_type': data['model_type'],
            'is_generic': data['is_generic'],
        }

    # Start the training in a new thread, passing the parameters to the training function
    t = threading.Thread(target=training, kwargs=kwargs)
    t.start()

    # Store the thread object and cancel event in the running_tasks and cancel_events dictionaries, respectively
    #running_tasks[(kwargs['wellid'], kwargs['servername'])] = t
    #cancel_events[(kwargs['wellid'], kwargs['servername'])] = cancel_event

    # Return a success response
    return JsonResponse({"message": "Training started successfully."})



@api_view(["POST"])
def check_training_status(request):

    servername = request.data["servername"]
    user_id = request.data["user_id"]

    try:
        # Check if there is any training in progress
        status_obj = TrainingStatus.objects.filter(servername=servername, user_id=user_id).latest('created_at')
        if status_obj.status == "in progress":
            return JsonResponse({"message": "Training is in progress"})
        else:
            # Fetch the most recent metrics from the TrainingMetrics table
            metrics_obj = status_obj.metrics.latest('created_at')
            #trained_model_obj = TrainedModel.objects.filter(servername=servername, wellid=wellid, user_id=user_id).latest('created_at')
            if metrics_obj.algo_name=="random-forest":
                return JsonResponse({
                    "algo_name": metrics_obj.algo_name,
                    "model_name": metrics_obj.model_name,
                    "pattern_ranges": metrics_obj.pattern_ranges,
                    "normal_ranges": metrics_obj.normal_ranges,
                    "model_description": metrics_obj.model_description,
                    "precision": metrics_obj.training_loss,
                    "accuracy": metrics_obj.training_accuracy,
                    "recall": metrics_obj.validation_loss,
                    "f1": metrics_obj.validation_accuracy,
                    "confusion_00": metrics_obj.training_loss_values,
                    "confusion_01": metrics_obj.training_accuracy_values,
                    "confusion_10": metrics_obj.validation_loss_values,
                    "confusion_11": metrics_obj.validation_accuracy_values,
                    "trained_at": metrics_obj.created_at,
                    "trained_at_ms": metrics_obj.created_at_ms,
                    "creation_source": metrics_obj.creation_source,
                    "user_id": status_obj.user_id,
                    "model_type": metrics_obj.model_type,
                    "inputs":metrics_obj.inputs,
                })
            else :  
                return JsonResponse({
                    "algo_name": metrics_obj.algo_name,
                    "model_name": metrics_obj.model_name,
                    "pattern_ranges": metrics_obj.pattern_ranges,
                    "normal_ranges":metrics_obj.normal_ranges,
                    "model_description": metrics_obj.model_description,
                    "training_loss": metrics_obj.training_loss,
                    "training_accuracy": metrics_obj.training_accuracy,
                    "validation_loss": metrics_obj.validation_loss,
                    "validation_accuracy": metrics_obj.validation_accuracy,
                    "training_loss_values": metrics_obj.training_loss_values,
                    "training_accuracy_values": metrics_obj.training_accuracy_values,
                    "validation_loss_values": metrics_obj.validation_loss_values,
                    "validation_accuracy_values": metrics_obj.validation_accuracy_values,
                    "trained_at": metrics_obj.created_at,
                    "trained_at_ms": metrics_obj.created_at_ms,
                    "creation_source": metrics_obj.creation_source,
                    "user_id": status_obj.user_id,
                    "model_type": metrics_obj.model_type,
                    "inputs":metrics_obj.inputs,
                })
    
    except TrainingStatus.DoesNotExist:
        return JsonResponse({"message": "No training has been started for this well"})
    except TrainingMetrics.DoesNotExist:
        return JsonResponse({"message": "Training has been completed but metrics have not been saved to the database"})
    except Exception as e:
        print("Error in check_training_status")
        print(e)
        return JsonResponse({"message": "Error in check_training_status"})


@api_view(["POST"])
def view_synthetic_data(request):
    if request.method=="POST":
        data = json.loads(request.body)

        servername = str(data["servername"])
        wellid = int(data["wellid"])
        start = int(data["start"])
        end = int(data["end"])
        samples = int(data["samples"])
        noise = float(data["noise"])
        tags = data["tags"]
        sample_no = int(data["sample_no"])

        # call the search logic method to retrieve data
        df = load_data(servername= servername, start=start, end=end, tags=tags, wellid=wellid)
        
        df = df.set_index("0")
        print("\n df original shape", df.shape)

        # call the processing method to process data 
        df, preprocessing_logs, status = preprocess(df=df, columns = tags)
        if status=='fail':
            return Response({'status':status, 'message':preprocessing_logs})
        
        # call noise and time methods 
        df_all = generate_synthetic_data(df=df, samples=samples, percentage=noise)

        df_sample = pd.DataFrame(df_all[sample_no])
        print(f"\n df all {sample_no}° element shape",df_sample.shape,"\n")

        # Convert dataframe to JSON string
        df_sample = df_sample.to_dict(orient='list')
    
        response = {'status': 'success', 'data': df_sample}
        return Response(response, status=st.HTTP_200_OK)

    return Response(response, status = st.HTTP_400_BAD_REQUEST) 


@api_view(["POST"])
def save_model(request):
    if request.method == 'POST':
        # Get the data from the request
        data = json.loads(request.body)
        servername = data['servername']

        # Get the most recent TrainingStatus and TrainingMetrics objects
        status_obj = TrainingStatus.objects.filter(servername=data['servername'], user_id=data['user_id']).latest('created_at')
        metrics_obj = status_obj.metrics.latest('created_at')
        algo_name = metrics_obj.algo_name

        print("------------------   Metrics and Status objects Received ----------------")

        # Create a new TrainedModel object
        trained_model = TrainedModel(
            servername=data['servername'],
            status=status_obj.status,
            created_at = metrics_obj.created_at, #datetime.now(), # use metrics_obj.created_at
            created_at_ms = metrics_obj.created_at_ms,
            model_name=metrics_obj.model_name,
            algo_name=metrics_obj.algo_name,
            pattern_ranges=metrics_obj.pattern_ranges,
            normal_ranges=metrics_obj.normal_ranges,
            model_description=metrics_obj.model_description,
            training_loss=metrics_obj.training_loss,
            training_accuracy=metrics_obj.training_accuracy,
            validation_loss=metrics_obj.validation_loss,
            validation_accuracy=metrics_obj.validation_accuracy,
            training_loss_values=metrics_obj.training_loss_values,
            training_accuracy_values=metrics_obj.training_accuracy_values,
            validation_loss_values=metrics_obj.validation_loss_values,
            validation_accuracy_values=metrics_obj.validation_accuracy_values,
            trained_at=datetime.now(),
            user_id=data['user_id'],
            model_type=metrics_obj.model_type,
            creation_source="user",
            last_trained=metrics_obj.last_trained,
            training_time=metrics_obj.training_time,
            execution_time=metrics_obj.execution_time,
            is_generic=metrics_obj.is_generic,
            inputs=metrics_obj.inputs,
        )

        # Save the TrainedModel object to the database
        trained_model.save()

        # Update the filename of the trained model
        if algo_name =="auto-encoder":
            unique_filename = f"model_{servername}_{metrics_obj.created_at_ms}.h5"
            model_file_path = os.path.join(".", unique_filename)
            model_id = trained_model.pk
            new_filename = f"model_{model_id}.h5"
            model_dir = "./models/"
            new_file_path = os.path.join(model_dir, new_filename)

        elif algo_name=="random-forest":
            unique_filename = f"model_{servername}_{metrics_obj.created_at_ms}.pkl"
            model_file_path = os.path.join(".", unique_filename)
            model_id = trained_model.pk
            new_filename = f"model_{model_id}.pkl"
            model_dir = "./models/"
            new_file_path = os.path.join(model_dir, new_filename)

        elif algo_name=="svc":
            unique_filename = f"model_{servername}_{metrics_obj.created_at_ms}.pkl"
            model_file_path = os.path.join(".", unique_filename)
            model_id = trained_model.pk
            new_filename = f"model_{model_id}.pkl"
            model_dir = "./models/"
            new_file_path = os.path.join(model_dir, new_filename)

        elif algo_name=="isolation-forest":
            unique_filename = f"model_{servername}_{metrics_obj.created_at_ms}.pkl"
            model_file_path = os.path.join(".", unique_filename)
            model_id = trained_model.pk
            new_filename = f"model_{model_id}.pkl"
            model_dir = "./models/"
            new_file_path = os.path.join(model_dir, new_filename)


        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Copy the trained model file
        shutil.copy(model_file_path, new_file_path)

        # Delete the original trained model file
        os.remove(model_file_path)

        # Set the file path in the TrainedModel object
        trained_model.file_path = new_file_path
        trained_model.save()

        # do the same for the scaler

        scaler_filename = f"scaler_{servername}_{metrics_obj.created_at_ms}.pkl"
        if os.path.exists(scaler_filename):
            scaler_file_path = os.path.join(".", scaler_filename)
            model_id = trained_model.pk
            new_filename = f"scaler_{model_id}.pkl"
            scaler_dir = "./scalers/"
            new_file_path = os.path.join(scaler_dir, new_filename)

            if not os.path.exists(scaler_dir):
                os.makedirs(scaler_dir)

            # Copy the trained model file
            shutil.copy(scaler_file_path, new_file_path)
            os.remove(scaler_file_path)

        # Return a response to the client
        response_data = {'message': 'Model saved successfully'}
        return HttpResponse(json.dumps(response_data), content_type='application/json')
    
    else:
        # Return an error response if the request method is not POST
        response_data = {'error': 'Invalid request method'}
        return HttpResponse(json.dumps(response_data), content_type='application/json', status=400)





""" --------------------------------------------    Backtesting  service  -------------------------------------------"""
from sources.src.upload_data import search_logic, load_influx_data
from sources.src.preprocess_data import backtesting_auto_encoder,backtesting_random_forest


def predict_probs(losses, threshold):
    probabilities = []
    for loss in losses:
        if loss > threshold:
            probabilities.append(loss*100)
        else:
            probabilities.append((1 - loss)*100)
    return np.array(probabilities)


import re

@api_view(['POST'])
def predict(request):

    data = json.loads(request.body)

    model_id = data['model_id'] # 1
    wellids = data['wellids']   # 2 
    wellid = int(wellids[0])     
    timestamp = data['timestamp']   # 3 
    servername = data['servername']  # 4 
    inputs_model = data['inputs_model'] # 5

    #algo_name = data['algo_name']
    print("------------------------- Request is received successfully ",data)

    # Get info from of the model object
    # model_obj = TrainedModel.objects.get(id=model_id)
    # algo_name = model_obj.algo_name
    # servername = model_obj.servername
    # inputs_model = model_obj.inputs
    # inputs_model = ast.literal_eval(inputs_model)
    # inputs_model.insert(0, '0')

    # Regex pattern to check if the path ends with .h5 or .pkl
    
    # pattern_h5 = re.compile(rf'./models/model_{model_id}\.h5$')
    # pattern_pkl = re.compile(rf'./models/model_{model_id}\.pkl$')

    # Define the paths based on model_id
    h5_path = f'./models/model_{model_id}.h5'
    pkl_path = f'./models/model_{model_id}.pkl'

    # Check if the files exist and match the patterns
    if os.path.exists(h5_path):
        print("------------------------- The path ends with .h5 and the file exists")
        model_path = h5_path
        algo_name = "auto-encoder"
    elif os.path.exists(pkl_path):
        print("------------------------- The path ends with .pkl and the file exists")
        model_path = pkl_path
        algo_name = "random-forest"
    else:
        print("The path does not match either .h5 or .pkl, or the file does not exist")


    # Load models
    try:
        if algo_name=="auto-encoder":
            model_path = f'./models/model_{model_id}.h5'
            loaded_model = tf.keras.models.load_model(model_path)

        elif algo_name=="random-forest" or algo_name=="isolation-forest":
            model_path=f'./models/model_{model_id}.pkl'
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
                print("------------------------- Model is Loaded Successfully.")
    except Exception as e:
        print(e)
        return JsonResponse("Error in Loading model: Model not Found", status=400, safe=False)
    


    # Load the saved scaler from the file
    scaler_path = f'./scalers/scaler_{model_id}.pkl'
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            loaded_scaler = pickle.load(f)
            print("------------------------- loaded scaler  ")


    # loading the historical  data
    try:
        #tags = model_obj.inputs
        #tags = ast.literal_eval(tags)
        #tag_list = [int(num) for num in tags]
        data = load_data(servername=servername, start=timestamp-(3600000*6), end=timestamp, tags= inputs_model, wellid=wellid)
        
    except Exception as e:
        print(e)
        return JsonResponse("Error in Loading Data", status=400, safe=False)

        
    if len(data)<10:
        return JsonResponse("Data is not enough to be processed: Try with a recent timestamp", status=400, safe=False)

    final_data = data

    if algo_name=="auto-encoder":
        try:
            inference_auto_encoder(loaded_scaler, final_data,loaded_model, servername, model_id, wellid, timestamp)
            return JsonResponse({"Predictions made successfully for model id": model_id,"wellid":wellid}, status=200)
        except Exception as e:
            print(e)
            return JsonResponse("Error in inference with Auto-Encoder.", status=400, safe=False)

    elif algo_name=="random-forest":
        try:
            inference_random_forest(loaded_scaler=loaded_scaler, final_data=final_data ,model=loaded_model, servername=servername, model_id=model_id,
                                wellid=wellid, timestamp=timestamp)
            return JsonResponse({"Predictions made successfully for model id": model_id,"wellid":wellid}, status=200)
        except Exception as e:
            print(e)
            return JsonResponse("Error in inference with Random-Forest.", status=400, safe=False)
        
    elif algo_name == "isolation-forest":
        try:
            inference_isolation_forest(final_data=final_data ,model=loaded_model, servername=servername,
                                        model_id=model_id, wellid=wellid, timestamp=timestamp)
            return JsonResponse({"Predictions made successfully for model id": model_id,"wellid":wellid}, status=200)
        except Exception as e:
            print(e)
            return JsonResponse("Error in inference with Isolation Forest.", status=400, safe=False)



""" --------------------------------------------   End of Backtesting  service  -------------------------------------------"""

def perform_backtest_async(**kwargs):
    servername = kwargs['servername']
    wellid = kwargs['wellid']
    trained_at_ms = kwargs['trained_at_ms']
    user_id = kwargs['user_id']
    backtest_results_path  = kwargs['backtest_results_path']

    try:
        # Update the TestingStatus entry to indicate the backtesting has started
        TestingStatus.objects.create(
            servername=servername,
            wellid=wellid,
            status="running",
            created_at_ms=trained_at_ms,
            trained_at_ms=trained_at_ms,  # Update the trained_at_ms field
            user_id=user_id,
            backtest_start_time=timezone.now()  # Set the start time
        )
        
        # Perform the backtesting calculations
        calculated_results = perform_backtest(**kwargs)

        # Update the TestingStatus entry to indicate the backtesting is completed
        #status_obj = TrainingStatus.objects.filter(wellid=wellid, servername=servername, user_id=user_id).latest('created_at')

        status_entry = TestingStatus.objects.filter(user_id=user_id, trained_at_ms=trained_at_ms).order_by('-backtest_start_time').first()
        status_entry.status = "completed"
        status_entry.save()


        # Write the calculated results to a CSV file
        pd.DataFrame(calculated_results).to_csv(backtest_results_path)

        return backtest_results_path
    except Exception as e:
        # Update the TestingStatus entry to indicate an error occurred
        status_entry = TestingStatus.objects.get(user_id=user_id, trained_at_ms=trained_at_ms)
        status_entry.status = "error"
        status_entry.save()

        return None


@csrf_exempt
def start_backtest(request):
    #global backtest_results, backtest_done
    
    if request.method == "POST":
        data = json.loads(request.body)

        algo_name = data["algo_name"]
        user_id = data["user_id"]
        trained_at_ms = data["trained_at_ms"]

        # Create a unique filename for the backtesting results CSV
        unique_filename = f'user_backtest_results_{user_id}_{trained_at_ms}.csv'
        backtest_results_path = os.path.join(tempfile.gettempdir(), unique_filename)

        if algo_name == "auto-encoder":
            kwargs ={
                'algo_name':data["algo_name"],
                'user_id': data["user_id"],  
                'servername': data["servername"],
                'wellid':data["wellid"],
                'start':data["start"],
                'end':data["end"],
                'trained_at_ms':data["trained_at_ms"],
                'columns':data["inputs"],
                'backtest_results_path':backtest_results_path
                } 
            
        elif algo_name == "random-forest":
            kwargs = {
                'algo_name':data["algo_name"],
                'user_id': data["user_id"],  
                'servername': data["servername"],
                'wellid':data["wellid"],
                'start':data["start"],
                'end':data["end"],
                'trained_at_ms':data["trained_at_ms"],
                'columns':data["inputs"],
                'backtest_results_path':backtest_results_path
            }

        elif algo_name == "svc":
            kwargs={
                'algo_name':data["algo_name"],
                'user_id': data["user_id"],  
                'servername': data["servername"],
                'wellid':data["wellid"],
                'start':data["start"],
                'end':data["end"],
                'trained_at_ms':data["trained_at_ms"],
                'columns':data["inputs"],
                'backtest_results_path':backtest_results_path
            }

        elif algo_name == "isolation-forest":
            kwargs = {
                'algo_name':data["algo_name"],
                'user_id': data["user_id"], 
                'servername': data["servername"],
                'wellid':data["wellid"],
                'start':data["start"],
                'end':data["end"],
                'trained_at_ms':data["trained_at_ms"],
                'columns':data["inputs"],
                'backtest_results_path':backtest_results_path
            }


        # Create a new thread for backtesting
        backtest_thread = threading.Thread(
            target=perform_backtest_async,
            kwargs=kwargs
        )
        backtest_thread.start()
    
        return JsonResponse({"message": "Backtesting started"})
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)


@api_view(["POST"])
def check_backtest_status(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_id = data["user_id"]
        trained_at_ms = data["trained_at_ms"]

        if not user_id:
            return JsonResponse({"error": "User ID is missing in the request body"}, status=400)

        try:
            # Query the TestingStatus model to get the latest status entry
            latest_status_entry = TestingStatus.objects.filter(user_id=user_id, trained_at_ms=trained_at_ms).order_by('-backtest_start_time').first()
            print("latest status: ",latest_status_entry)

            if latest_status_entry:
                # Get the backtest start time from the TestingStatus entry
                backtest_start_time = latest_status_entry.backtest_start_time

                # Check if the status is 'running'
                if latest_status_entry.status == "running":
                    # Retrieve and include the progress in the response
                    return JsonResponse({"status": "running", "start_time": backtest_start_time}, status=200)
                

                # Check if the status is 'completed'
                elif latest_status_entry.status == "completed":
                    unique_filename = f'user_backtest_results_{user_id}_{trained_at_ms}.csv'
                    backtest_results_path = os.path.join(tempfile.gettempdir(), unique_filename)

                    if os.path.exists(backtest_results_path):
                        try:
                            # Read the CSV file using pandas
                            df = pd.read_csv(backtest_results_path)

                            # Create an empty dictionary to store the transformed data
                            result = {}

                            # Get the column names except for '0'
                            columns = df.columns[df.columns != '0']

                            # Iterate over each column in the dataframe
                            for column in columns:
                                # Get the values from the column and convert them to a list
                                values = df[column].values.tolist()

                                values = [float(vs) for vs in values]
                                # Create a list of timestamps based on the index values
                                timestamps = [int(ts) for ts in df['0']]
                                # Create a list of lists combining timestamps and values
                                data = [[timestamps[i], values[i]] for i in range(len(values))]
                                # Assign the list of data to the column key in the result dictionary
                                result[column] = data

                            # Delete the CSV file after processing
                            os.remove(backtest_results_path)

                            return JsonResponse(result, status=200, safe=False)
                        except Exception as e:
                            return JsonResponse({"error": "Error while processing CSV file"}, status=500)
                    else:
                        return JsonResponse({"message": "Backtesting results not available"}, status=404)

            else:
                return JsonResponse({"message": "Backtesting status not found"}, status=404)

        except Exception as e:
            print(f"Error details: {str(e)}")
            return JsonResponse({"error": "Error while processing backtesting status"}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)


"""----------------------------- Testing Saved Models on Historical Data ----------------------------"""


@api_view(["POST"])
def start_backtest_admin(request):    
    if request.method == "POST":
        data = json.loads(request.body)
        # from it we will pull of info
        model_id = data['model_id']
        user_id = data["user_id"]

        # we need those bcz we extract them from the model_id object
        servername = data['servername']
        wellid = data['wellid']
        start = data['start']
        end = data['end']

        # Get info from of the model object
        model_obj    = TrainedModel.objects.get(id=model_id)
        algo_name    = model_obj.algo_name
        inputs_model = model_obj.inputs
        inputs_model = ast.literal_eval(inputs_model)
        inputs_model = [int(num) for num in inputs_model]
        print("input models before loading data\n ",inputs_model)

        # Create a unique filename for the backtesting results CSV
        unique_filename = f'user_backtest_results_{model_id}_{wellid}.csv'
        backtest_results_path = os.path.join(tempfile.gettempdir(), unique_filename)

        if algo_name == "auto-encoder":
            kwargs ={
                'model_id':model_id,
                'algo_name':algo_name,
                'user_id': user_id,  
                'servername': servername,
                'wellid':wellid,
                'start':start,
                'end':end,
                'columns':inputs_model,
                'backtest_results_path':backtest_results_path
                } 

        # Create a new thread for backtesting
        backtest_thread = threading.Thread(
            target=perform_backtest_async_admin,
            kwargs=kwargs
        )
        backtest_thread.start()
    
        return JsonResponse({"message": "Backtesting started"})
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)



def perform_backtest_async_admin(**kwargs):
    servername = kwargs['servername']
    wellid = kwargs['wellid']
    user_id = kwargs['user_id']
    backtest_results_path  = kwargs['backtest_results_path']

    try:
        # Update the TestingStatus entry to indicate the backtesting has started
        TestingStatus.objects.create(
            servername=servername,
            wellid=wellid,
            status="running",
            user_id=user_id,
            backtest_start_time=timezone.now()  # Set the start time
        )
        # Perform the backtesting calculations
        calculated_results = perform_backtest_admin(**kwargs)
        status_entry = TestingStatus.objects.filter(user_id=user_id).order_by('-backtest_start_time').first()
        status_entry.status = "completed"
        status_entry.save()

        # Write the calculated results to a CSV file
        pd.DataFrame(calculated_results).to_csv(backtest_results_path)

        return backtest_results_path
    except Exception as e:
        # Update the TestingStatus entry to indicate an error occurred
        status_entry = TestingStatus.objects.get(user_id=user_id)
        status_entry.status = "error"
        status_entry.save()

        return None



@api_view(["POST"])
def check_backtest_status_admin(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_id  = data["user_id"]
        model_id = data["model_id"]
        wellid   = data["wellid"]

        if not user_id:
            return JsonResponse({"error": "User ID is missing in the request body"}, status=400)

        try:
            # Query the TestingStatus model to get the latest status entry
            latest_status_entry = TestingStatus.objects.filter(user_id=user_id).order_by('-backtest_start_time').first()
            print("latest status: ",latest_status_entry)

            if latest_status_entry:
                backtest_start_time = latest_status_entry.backtest_start_time

                # Check if the status is 'running'
                if latest_status_entry.status == "running":
                    return JsonResponse({"status": "running", "start_time": backtest_start_time}, status=200)
                
                # Check if the status is 'completed'
                elif latest_status_entry.status == "completed":
                    unique_filename = f'user_backtest_results_{model_id}_{wellid}.csv'
                    backtest_results_path = os.path.join(tempfile.gettempdir(), unique_filename)

                    if os.path.exists(backtest_results_path):
                        try:
                            # Read the CSV file using pandas
                            df = pd.read_csv(backtest_results_path)
                            result = {}
                            columns = df.columns[df.columns != '0']

                            for column in columns:
                                values = df[column].values.tolist()

                                values = [float(vs) for vs in values]
                                timestamps = [int(ts) for ts in df['0']]
                                data = [[timestamps[i], values[i]] for i in range(len(values))]
                                result[column] = data

                            os.remove(backtest_results_path)

                            return JsonResponse(result, status=200, safe=False)
                        except Exception as e:
                            return JsonResponse({"error": "Error while processing CSV file"}, status=500)
                    else:
                        return JsonResponse({"message": "Backtesting results not available"}, status=404)

            else:
                return JsonResponse({"message": "Backtesting status not found"}, status=404)

        except Exception as e:
            print(f"Error details: {str(e)}")
            return JsonResponse({"error": "Error while processing backtesting status"}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)




"""-------------------------- End of Backtest for Admins ----------------------------"""

@api_view(["DELETE"])
def delete_saved_model(request):
    if request.method == 'DELETE':
        # Get the data from the request
        data = json.loads(request.body)
        
        model_ids = data.get("model_ids", [])
        
        # Check if model_ids is a list
        if not isinstance(model_ids, list):
            response_data = {'error': 'Invalid model_ids format'}
            return JsonResponse(response_data, status=400)

        deleted_models = []
        not_found_models = []
        
        for model_id in model_ids:
            try:
                model = TrainedModel.objects.get(id=model_id)
                model.delete()
                deleted_models.append(model_id)
            except TrainedModel.DoesNotExist:
                not_found_models.append(model_id)

        response_data = {'deleted_models': deleted_models, 'not_found_models': not_found_models}
        return JsonResponse(response_data)
    else:
        # Return an error response if the request method is not DELETE
        response_data = {'error': 'Invalid request method'}
        return JsonResponse(response_data, status=400)






@api_view(["GET"])
def server_capacity(request):
    """
    Return the CPU and RAM % of  usage.    
    """
    if request.method=="GET":
        # RAM usage
        ram_usage  = psutil.virtual_memory()[2]
        # CPU usage
        cpu_usage  = psutil.cpu_percent(4)
        response = {"ram_usage":ram_usage, "cpu_usage":cpu_usage}
        return JsonResponse(response, safe=False)

    return Response(response, status = st.HTTP_400_BAD_REQUEST) 



@csrf_exempt
def get_saved_models(request):
    if request.method == 'GET':
        # Get all saved models
        saved_models = TrainedModel.objects.all().order_by('-trained_at')

        # Create a list of dictionaries containing the model information
        model_list = []
        for model in saved_models:
            model_dict = {
                'id': model.id,
                'servername': model.servername,
                'algo_name':model.algo_name,
                'pattern_ranges':model.pattern_ranges,
                'normal_ranges':model.normal_ranges,
                'status': model.status,
                'created_at_ms': model.created_at_ms,
                'model_name': model.model_name,
                'model_description': model.model_description,
                'training_loss': model.training_loss,
                'training_accuracy': model.training_accuracy,
                'validation_loss': model.validation_loss,
                'validation_accuracy': model.validation_accuracy,
                'training_loss_values': model.training_loss_values,
                'training_accuracy_values': model.training_accuracy_values,
                'validation_loss_values': model.validation_loss_values,
                'validation_accuracy_values': model.validation_accuracy_values,
                'trained_at': model.trained_at.strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': model.user_id,
                'model_type': model.model_type,
                'creation_source': model.creation_source,
                'inputs': model.inputs,
                'is_generic': model.is_generic,
                'last_trained': model.last_trained.strftime('%Y-%m-%d %H:%M:%S') if model.last_trained else None,
                'training_time': model.training_time,
                'execution_time': model.execution_time,
            }
            model_list.append(model_dict)

        # Return a JSON response containing the list of models
        response_data = {'models': model_list}
        return JsonResponse(response_data)
    else:
        # Return an error response if the request method is not GET
        response_data = {'error': 'Invalid request method'}
        return JsonResponse(response_data, status=400)


@api_view(["POST"])
def get_saved_models_greater_than(request):
    if request.method == 'POST':

        # Get the model ID and servername from the POST request data
        data = json.loads(request.body)
        model_id = data['model_id']
        servername = data['servername']

        # Get all saved models with ID greater than the specified ID and matching the servername
        saved_models = TrainedModel.objects.filter(id__gt=model_id, servername=servername).order_by('-trained_at')
        

        # Create a list of dictionaries containing the model information
        model_list = []
        for model in saved_models:
            inputs_list = ast.literal_eval(model.inputs) if model.inputs else []

            model_dict = {
                'id': model.id,
                'servername': model.servername,
                'algo_name':model.algo_name,
                'pattern_ranges':model.pattern_ranges,
                'normal_ranges':model.normal_ranges,
                'status': model.status,
                'created_at': model.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': model.model_name,
                'model_description': model.model_description,
                'training_loss': model.training_loss,
                'training_accuracy': model.training_accuracy,
                'validation_loss': model.validation_loss,
                'validation_accuracy': model.validation_accuracy,
                'training_loss_values': model.training_loss_values,
                'training_accuracy_values': model.training_accuracy_values,
                'validation_loss_values': model.validation_loss_values,
                'validation_accuracy_values': model.validation_accuracy_values,
                'trained_at': model.trained_at.strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': model.user_id,
                'model_type': model.model_type,
                'creation_source': model.creation_source,
                'inputs': inputs_list,
                'is_generic': model.is_generic,
                'last_trained': model.last_trained.strftime('%Y-%m-%d %H:%M:%S') if model.last_trained else None,
                'training_time': model.training_time,
                'execution_time': model.execution_time,
            }
            model_list.append(model_dict)

        # Return a JSON response containing the list of models
        response_data = {'models': model_list}
        return JsonResponse(response_data)
    else:
        # Return an error response if the request method is not POST
        response_data = {'error': 'Invalid request method'}
        return JsonResponse(response_data, status=400)





"""----------------------------------- Integrating Manual Models -------------------------------"""

from sources.src.utilities import manual_pipeline
from sklearn.preprocessing import QuantileTransformer

@api_view(["POST"])
def predict_manual(request):

    data = json.loads(request.body)

    servername = data["servername"]
    wellid = data["wellid"]
    timestamp = data["timestamp"]

    models = {
        "44":"3001",
        "46":"3003",
        "47":"3004",
        "48":"3005",
        "49":"3006",
        "50":"3013",
        "51":"3008",
        "52":"3007",
        "53":"3009",
        "54":"3010", 
        "55":"3011",
        }
    
    # fetching data
    try:
        input_test = manual_pipeline(wellid=wellid, start=timestamp-(3600000*6), end=timestamp)
        print(f"---------------------- data is loaded successfully and has a length of {input_test.shape}")
    except Exception as e:
        print(e)

    input_test = np.array(input_test)
    input_test = input_test.reshape(-1,1)
    scaler = QuantileTransformer()

    try:
        input_test = scaler.fit_transform(input_test)
        print(f"---------------------- data is scaled successfully ")
    except Exception as e:
        print(e)
    
    # for i in range(0, input_test.shape[0],720): 
    #     input_test[i:i+720] = scaler.fit_transform(input_test[i:i+720])


    current_directory = os.getcwd()
    contents = os.listdir(current_directory)

    print("Contents of the current directory:")
    for item in contents:
        print(item)


    num_elements = len(input_test)
    sequence_length = 720
    dir_model = "./manual_models/" 

    results = {}

    # start looping
    for j in range(len(models)):
        print(f"we are processing model {list(models.values())[j]}")
        probs=[]
        model_path = dir_model+list(models.keys())[j]+".pkl"
        loaded_model = joblib.load(model_path)
        print(f"model {list(models.values())[j]} was loaded ")

        # looping over the dataset with shifting window 
        for start, stop in zip(range(0, num_elements-sequence_length, 15), range(sequence_length, num_elements, 15)):            
            input_=np.array(input_test[start:stop])
            input_=input_.reshape(1,-1)
            res_list = loaded_model.predict_proba(input_)[0]
            probs.append(res_list[1])
            print("prediction is ",res_list[1])

        #added section
        probs_rl =pd.DataFrame(probs, columns=["probs"]).rolling(10).mean()
        print("probs_rl",probs_rl)
        probs_f = list(probs_rl["probs"][-1:]) 
        results[list(models.values())[j]] = probs_f

    """ ----------------------------- Saving the final result -------------------------------"""
    # check if the folder exists
    result_path = f"/mnt/{servername}/mlPrediction/Manual_result_{wellid}_{timestamp}.json"
    try:
      outfile = open(result_path, "w") # used in prod
      json.dump(results, outfile)
      outfile.close()
    except IOError as e:
      print("the mode of the file is incorect")
      print(e)
    else:
      print("the data has been saved to the file successfuly") 

    response_data = {'response_data': results}
    return JsonResponse(response_data, status=400)
