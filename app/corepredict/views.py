from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework import status as st
from rest_framework.decorators import api_view
from datetime import datetime
from django.http import HttpResponse
from django.http import JsonResponse, HttpResponseBadRequest
import tempfile
from django.utils import timezone
from django.http import JsonResponse
import threading
from django.utils.timezone import now
import pickle
import os
import time
import json
import pandas as pd
import numpy as np
from .models import  TrainingMetrics, TrainingStatus, TrainedModel, TestingStatus
from .main_predict import main_predict
import joblib
from sklearn.ensemble import RandomForestRegressor
import joblib
from .main_predict import load_influx_intelift, next_month_start
from sources.src.upload_data import load_data
import ast




@api_view(["POST"])
def feature_selection_correlation(request):
    # Parse the JSON data from the request body
    try:
        data = json.loads(request.body)
        servername = data['servername']
        start = data['start']
        end = data['end']
        tags = data['tags']
        wellid = data['wellid']
        y = data.get('target_tag')  # Target variable
        corr_method = data.get('corr_method', 'spearman')  # Default to Spearman if not specified

        if not y:
            return HttpResponseBadRequest("The 'target_tag' parameter is required.")
        
        valid_methods = ['pearson', 'spearman', 'kendall']
        
        if corr_method not in valid_methods:
            return HttpResponseBadRequest(f"The 'corr_method' must be one of {valid_methods}")

        # load the requested Historical data
        all_tags = tags + [y]
        df = load_data(servername=servername, start=start, end=end, tags=all_tags, wellid=wellid)
        df = df.set_index('0')
        print("--------------------- data is loaded successfully")





        # Calculate correlation using the specified method
        corr = df.corr(method=corr_method)[[y]].dropna(axis=0)
        print("--------------------- correlation is computed")
        corr['Correlation'] = corr[y].apply(lambda x: "Positive" if x > 0 else "Negative")
        #corr[y] = corr[y].abs()  # Make all correlation values positive
        corr.sort_values(by=y, ascending=False, inplace=True)

        # Convert the DataFrame to a dictionary for JSON response
        corr_dict = corr.to_dict()

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'message': 'Correlation calculated successfully.', 'correlation': corr_dict})




""" -------------------- Start of Feature Selection using RF Method -------------------------- """





@api_view(["POST"])
def feature_selection_importance(request):
    # Parse the JSON data in the request body to a dictionary
    data = json.loads(request.body)

    start = data['start']
    end = data['end']
    tags = data['tags']
    wellid = data['wellid']
    y = data.get('target_tag')  # Assuming 'target_tag' is provided in the request data

    # Load df
    df = load_influx_intelift(start=start, end=end, tags=tags, wellid=wellid)

    # Prepare the features and target variable
    x = list(df.columns.difference([y]))
    X = df[x]
    Y = df[y].values

    # Fit RandomForestRegressor
    random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest_regressor.fit(X, Y)

    # Calculate feature importances
    feature_importances = random_forest_regressor.feature_importances_
    print("feature_importances", feature_importances)
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = [(X.columns[i], feature_importances[i]) for i in sorted_indices]

    # Convert sorted features to a format suitable for JSON response
    feature_importance_dict = {feature: importance for feature, importance in sorted_features}

    # Return a success response
    return JsonResponse({"message": "Feature importances calculated successfully.", "feature_importances": feature_importance_dict})

# Make sure to define load_influx_intelift function and include necessary imports for it to work.


""" -------------------------   End of Feature Selection using RF Method -------------------------- """

from .training import training

# A dictionary to store the running training tasks
running_tasks = {}
cancel_events = {}

@api_view(["POST"])
def start_training(request):
    # Parse the JSON data in the request body to a dictionary
    data = json.loads(request.body)

    cancel_event = threading.Event()
    # Extract the parameters for Prediction Task
    kwargs = {
            'servername': data['servername'],
            'pattern_ranges': data['pattern_ranges'],
            'tags': data['tags'],
            'target_column':data['target_column'],
            'model_name': data['model_name'],
            'model_description': data['model_description'],
            'user_id': data['user_id'],
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

            return JsonResponse({
                    "model_name": metrics_obj.model_name,
                    "model_description": metrics_obj.model_description,
                    "trained_at": metrics_obj.created_at,
                    "trained_at_ms": metrics_obj.created_at_ms,
                    "r2":metrics_obj.performance,
                    "mse":metrics_obj.mse,
                    "y_scaler_value":metrics_obj.y_scaler_value,
                    "creation_source": metrics_obj.creation_source,
                    "user_id": status_obj.user_id,
                    "inputs":metrics_obj.inputs,
                    "target_column":metrics_obj.target_column,
                })
        
    except TrainingStatus.DoesNotExist:
        return JsonResponse({"message": "No training has been started for this well"})
    except TrainingMetrics.DoesNotExist:
        return JsonResponse({"message": "Training has been completed but metrics have not been saved to the database"})
    except Exception as e:
        print("Error in check_training_status")
        print(e)
        return JsonResponse({"message": "Error in check_training_status"})


"""------------------------------------------- Backtesting Functions -------------------------------------------"""


def perform_backtest(**kwargs):
    print("------------------------- backtesting started")

    servername = kwargs["servername"]
    wellid = kwargs["wellid"]
    inputs = kwargs["columns"]
    trained_at = kwargs["trained_at_ms"]
    start = kwargs["start"]
    end = kwargs["end"]
    y_scaler_value = kwargs["y_scaler_value"]


    unique_filename = f"model_{servername}_{trained_at}.pkl"
    model_file_path = os.path.join(".", unique_filename)


    scaler_filename = f"scaler_{servername}_{trained_at}.pkl"
    scaler_file_path = os.path.join(".", scaler_filename)


    # Load the saved scaler from the file
    with open(scaler_file_path, 'rb') as f:
        loaded_scaler = pickle.load(f)
        print("------------------------- The scaler file loaded successfully.")

    # load the model
    try:
        with open(model_file_path, 'rb') as file:
            loaded_model = pickle.load(file)
            print("------------------------- The model file loaded successfully.")
    except Exception as e:
        print("Error in Loading model")
    
    # load the requested Historical data
    try:
        df = load_data(servername=servername, start=start, end=end, tags=inputs, wellid=wellid)
        print('------------------------- Historical Data is loaded successfully')
    except Exception as  e:
        print(e)
        print("Error in Loading Historical Data")


    df = df.set_index('0')

    test = df
    test_ = test.copy()

    try:
        # Ensure the columns in the same order as during training
        cols = test.columns
        indx = test.index
        print("------------------------- just starting the scaling step ")
        # Transform the test set using the loaded scaler
        test_scaled = loaded_scaler.transform(test)
        print("------------------------- passed the scaling step !")
        test_scaled = pd.DataFrame(test_scaled, columns=cols, index=indx)

        # Predict using the loaded model
        predictions = loaded_model.predict(test_scaled)
        print("------------------------- passed the predicting step !")

        
        # Denormalize the predictions
        test_['prediction'] = predictions * y_scaler_value
        
        print("------------------------- Results of backtesting are:\n", test_)
    except Exception as e:
        print("\nError in Backtesting (predicting) process")
        print(e)
        
    #converting the df_results requested by abdel
    test_ = pd.DataFrame(test_)

    return test_

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
        status_entry = TestingStatus.objects.filter(user_id=user_id, trained_at_ms=trained_at_ms).order_by('-backtest_start_time').first()
        status_entry.status = "completed"
        status_entry.save()


        # Write the calculated results to a CSV file
        pd.DataFrame(calculated_results).to_csv(backtest_results_path)
        print(f" --------------  test data is saved locally under {backtest_results_path}")

        return backtest_results_path
    except Exception as e:
        # Update the TestingStatus entry to indicate an error occurred
        status_entry = TestingStatus.objects.get(user_id=user_id, trained_at_ms=trained_at_ms)
        status_entry.status = "error"
        status_entry.save()

        return None


@api_view(["POST"])
def start_backtest(request):
    
    if request.method == "POST":
        data = json.loads(request.body)
        user_id = data["user_id"]
        trained_at_ms = data["trained_at_ms"]

        # Create a unique filename for the backtesting results CSV
        unique_filename = f'user_backtest_prediction_{user_id}_{trained_at_ms}.csv'
        backtest_results_path = os.path.join(tempfile.gettempdir(), unique_filename)

        kwargs ={
            "user_id": data["user_id"],  
            "servername": data["servername"],
            "wellid":data["wellid"],
            "start":data["start"],
            "end":data["end"],
            "trained_at_ms":data["trained_at_ms"],
            "columns":data["inputs"],
            "y_scaler_value":data["y_scaler_value"],
            "backtest_results_path":backtest_results_path
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
def check_backtesting_status(request):
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
                backtest_start_time = latest_status_entry.backtest_start_time

                # Check if the status is 'running'
                if latest_status_entry.status == "running":
                    return JsonResponse({"status": "running", "start_time": backtest_start_time}, status=200)
                
                # Check if the status is 'completed'
                elif latest_status_entry.status == "completed":
                    unique_filename = f'user_backtest_prediction_{user_id}_{trained_at_ms}.csv'
                    backtest_results_path = os.path.join(tempfile.gettempdir(), unique_filename)

                    if os.path.exists(backtest_results_path):
                        try:
                            df = pd.read_csv(backtest_results_path, index_col=0)
                            print("df columns", df.columns)
                            result = {}
                            columns = df.columns
                            # Iterate over each column in the dataframe
                            for column in columns:
                                values = df[column].values.tolist()
                                values = [float(vs) for vs in values]
                                timestamps = [int(ts) for ts in df.index]
                                data = [[timestamps[i], values[i]] for i in range(len(values))]
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



import shutil
@api_view(["POST"])
def save_model(request):
    if request.method == 'POST':
        # Get the data from the request
        data = json.loads(request.body)
        wellid = data['wellid']
        servername = data['servername']

        # Get the most recent TrainingStatus and TrainingMetrics objects
        status_obj = TrainingStatus.objects.filter(servername=data['servername'], user_id=data['user_id']).latest('created_at')
        metrics_obj = status_obj.metrics.latest('created_at')

        print("------------------   Metrics and Status objects Received ----------------")

        # Create a new TrainedModel object
        trained_model = TrainedModel(
            servername=data['servername'],
            wellid=data['wellid'],
            status=status_obj.status,
            created_at = metrics_obj.created_at, #datetime.now(), # use metrics_obj.created_at
            created_at_ms = metrics_obj.created_at_ms,
            model_name=metrics_obj.model_name,
            start = metrics_obj.start,
            end=metrics_obj.end,
            model_description=metrics_obj.model_description,
            performance=metrics_obj.performance,
            mse=metrics_obj.mse,
            y_scaler_value=metrics_obj.y_scaler_value,
            trained_at=datetime.now(),
            user_id=data['user_id'],
            creation_source="user",
            last_trained=metrics_obj.last_trained,
            training_time=metrics_obj.training_time,
            execution_time=metrics_obj.execution_time,
            is_generic=metrics_obj.is_generic,
            inputs=metrics_obj.inputs,
            target_column=metrics_obj.target_column,
            file_path=None,  # initialize the file path to None,
        )

        # Save the TrainedModel object to the database
        trained_model.save()

        # Update the filename of the trained model
        unique_filename = f"model_{servername}_{metrics_obj.created_at_ms}.pkl"
        model_file_path = os.path.join(".", unique_filename)
        model_id = trained_model.pk
        new_filename = f"model_pr_{model_id}.pkl"
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
            new_filename = f"scaler_pr_{model_id}.pkl"
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


@api_view(["GET"])
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
                'wellid': model.wellid,
                'start':model.start,
                'end':model.end,
                'status': model.status,
                'created_at_ms': model.created_at_ms,
                'model_name': model.model_name,
                'model_description': model.model_description,
                'model_inputs':model.inputs,
                'target_tag':model.target_column,
                'performance':model.performance,
                'mse': model.mse,
                'y_scaler_value':model.y_scaler_value,
                'trained_at': model.trained_at.strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': model.user_id,
                'creation_source': model.creation_source,
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




"""------------------------------- Get Saved Models Greater than ----------------------------"""


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
            model_dict = {
                'id': model.id,
                'servername': model.servername,
                'wellid': model.wellid,
                'start':model.start,
                'end':model.end,
                'status': model.status,
                'created_at_ms': model.created_at_ms,
                'model_name': model.model_name,
                'model_description': model.model_description,
                'model_inputs':model.inputs,
                'target_tag':model.target_column,
                'performance':model.performance,
                'mse': model.mse,
                'y_scaler_value':model.y_scaler_value,
                'trained_at': model.trained_at.strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': model.user_id,
                'creation_source': model.creation_source,
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


"""--------------------------------------- Delete Saved Models -----------------------------------"""



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




@api_view(['POST'])
def predict(request):
    if request.method=="POST":

        data = json.loads(request.body)
        print("------------------------- Request is received successfully ",data)
        model_id = data['model_id']
        wellids = data['wellids']
        wellid = int(wellids[0])

        timestamp = data['timestamp']
        # Get info from of the model object
        model_obj = TrainedModel.objects.get(id=model_id)
        servername = model_obj.servername
        y_scaler_value = model_obj.y_scaler_value
        inputs_model = model_obj.inputs
        inputs_model = ast.literal_eval(inputs_model)
        inputs_model.insert(0, '0')

        # Load models
        try:
            model_path=f'./models/model_pr_{model_id}.pkl'
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
            print("------------------------- Model is Loaded Successfully.")
        except Exception as e:
            print(e)
            return JsonResponse("Error in Loading model: Model not Found", status=400, safe=False)
    


        # Load the saved scaler from the file
        scaler_path = f'./scalers/scaler_pr_{model_id}.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                loaded_scaler = pickle.load(f)
                print("------------------------- loaded scaler  ")


        # loading the historical  data
        try:
            tags = model_obj.inputs
            tags = ast.literal_eval(tags)
            tag_list = [int(num) for num in tags]
            data = load_data(servername=servername, start=timestamp-(3600000*6), end=timestamp, tags= tag_list, wellid=wellid)
        except Exception as e:
            print(e)
            return JsonResponse("Error in Loading Data", status=400, safe=False)

            
        if len(data)<10:
            return JsonResponse("Data is not enough to be processed: Try with a recent timestamp", status=400, safe=False)

        input_data = data

        input_data = input_data.set_index('0')

        input_ = input_data.copy() 

        try:
            # Ensure the columns in the same order as during training
            cols = input_data.columns
            indx = input_data.index
            # Transform the test set using the loaded scaler
            input_scaled = loaded_scaler.transform(input_)
            input_scaled = pd.DataFrame(input_scaled, columns=cols, index=indx)
            # Predict using the loaded model
            predictions = loaded_model.predict(input_scaled)        
            input_data['Predictions'] = predictions * y_scaler_value
            print("Results of prediction are:\n", input_data)
        except Exception as e:
            print("\nError in predicting process")
            print(e)


        output_df = input_data
        output_df = output_df.reset_index().rename(columns={'0': 'Timestamp'})[['Timestamp', 'Predictions']]

        path_to_save_in = f'/mnt/{servername}/mlPrediction/prediction_{model_id}_{wellid}_{timestamp}.csv'
        output_df.to_csv(path_to_save_in, index=False)
        print(f'------------------------- File is saved  succeffully at {path_to_save_in} \n')

        return JsonResponse({"Predictions made successfully for model id": model_id,"wellid":wellid}, status=200)
    else: 
        return JsonResponse("Error in inference with Random-Forest.", status=400, safe=False)
        