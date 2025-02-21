from sources.src.main import main, main_rf, main_if, main_svc
from .models import  TrainingMetrics, TrainingStatus
from django.utils.timezone import now
import tensorflow as tf
import pandas as pd
import json
import os
import pickle
import time
import json



def training(**kwargs):
    servername = kwargs['servername']
    algo_name = kwargs['algo_name']
    user_id = kwargs['user_id']
    
    # Save the initial training status as "in progress"
    status_obj = TrainingStatus(
        servername=servername,
        status="in progress",
        created_at=now(),
        created_at_ms=int(time.time() * 1000),  # Convert current time to milliseconds
        user_id=user_id
    )
    status_obj.save()

    if algo_name == "auto-encoder":

        # extract the kwargs for the auto-encoder
        pattern_ranges = kwargs['pattern_ranges']
        normal_ranges = kwargs['pattern_ranges']
        tags = kwargs['tags']
        batch_size = kwargs['batch_size']
        encoding_dim = kwargs['encoding_dim']
        epochs = kwargs['epochs']
        model_name = kwargs['model_name']
        model_description = kwargs['model_description']
        model_type = kwargs['model_type']
        is_generic = kwargs['is_generic']

        try:
            # Train the auto-encoder model
            response = main(servername=str(servername),
                                        pattern_ranges=list(pattern_ranges),
                                        encoding_dim=int(encoding_dim),
                                        batch_size=int(batch_size),
                                        epochs=int(epochs),
                                        tags=list(tags))
            
            print("------------------------- Training is complete")
            status = "complete"
            model = response['model']
            scaler = response["scaler"]

        except Exception as e:
            print("Error in the main function")
            status = "failed"

    elif algo_name == "random-forest":
        # extract the kwargs for the auto-encoder
        pattern_ranges = kwargs['pattern_ranges']
        normal_ranges = kwargs['normal_ranges']
        tags = kwargs['tags']
        model_name = kwargs['model_name']
        model_description = kwargs['model_description']
        model_type = kwargs['model_type']
        is_generic = kwargs['is_generic']

        try:
            # Train the random-forest model
            response = main_rf( 
                                servername=str(servername),
                                pattern_ranges=list(pattern_ranges), 
                                normal_ranges=list(normal_ranges), 
                                tags=list(tags),
                                )
            
            print("------------------------- Training is complete ")
            status = "complete"
            model = response['model']
            scaler = response["scaler"]

        except Exception as e:
            print("Error in the main function")
            print(e)

    elif algo_name == "svc":
        # extract the kwargs for the auto-encoder
        pattern_ranges = kwargs['pattern_ranges']
        normal_ranges = kwargs['normal_ranges']
        tags = kwargs['tags']
        model_name = kwargs['model_name']
        model_description = kwargs['model_description']
        model_type = kwargs['model_type']
        is_generic = kwargs['is_generic']

        try:
            # Train the random-forest model
            response = main_svc( 
                                servername=str(servername),
                                pattern_ranges=list(pattern_ranges), 
                                normal_ranges=list(normal_ranges), 
                                tags=list(tags),
                                )
            
            print("------------------------- Training is complete ")
            status = "complete"
            model = response['model']
            scaler = response["scaler"]

        except Exception as e:
            print("Error in the main function")
            print(e)

    # -- u add the next algorithm
    elif algo_name == "isolation-forest":
        tags = kwargs['tags']
        samples = kwargs['samples']
        noise = kwargs['noise']
        n_trees = kwargs['n_trees']
        max_samples = kwargs['max_samples']
        contamination = kwargs['contamination']
        model_name = kwargs['model_name']
        model_description = kwargs['model_description']
        model_type = kwargs['model_type']
        is_generic = kwargs['is_generic']

        try:
            # Train the auto-encoder model
            response = main_if(servername=str(servername),
                                       samples=int(samples), noise=float(noise),
                                       n_trees=int(n_trees), max_samples=str(max_samples),
                                       contamination=float(contamination), tags=list(tags))
            
            print("------------------------- Training is complete")
            status = "complete"
            model = response['model']
            scaler = response["scaler"]

        except Exception as e:
            print("Error in the main function")
            print(e)
            status = "failed"

    try:
        # Update the training status based on the outcome of the training
        status_obj = TrainingStatus.objects.filter(servername=servername, user_id=user_id).latest('created_at')
        status_obj.status = status
        status_obj.save()
        print("-------------------------  Status updated in db successfully ")
    except Exception as e:
        print("Error in updating status")
        print(e)

    # Save the metrics in the TrainingMetrics model
    try:
        status_obj = TrainingStatus.objects.filter(servername=servername, user_id=user_id).latest('created_at')
        metrics_created_at = now().strftime('%Y-%m-%d %H:%M:%S')
        metrics_created_at_ms = int(time.time() * 1000)  # Convert current time to milliseconds
        metrics = TrainingMetrics(
            algo_name = algo_name,
            pattern_ranges = pattern_ranges,
            normal_ranges = normal_ranges,
            training_status = status_obj,
            model_name = model_name,
            model_description = model_description,
            is_generic = is_generic,
            inputs = str(tags),
            training_loss = response["training_loss"],
            training_accuracy = response["training_accuracy"],
            validation_loss = response["validation_loss"],
            validation_accuracy = response["validation_accuracy"],
            training_loss_values = str(response["training_loss_values"]),
            training_accuracy_values = str(response["training_accuracy_values"]),
            validation_loss_values = str(response["validation_loss_values"]),
            validation_accuracy_values = str(response["validation_accuracy_values"]),
            created_at = metrics_created_at,
            created_at_ms = metrics_created_at_ms,
            model_type = model_type,
            creation_source = "user",
            last_trained = status_obj.created_at,
            training_time = 0,
            execution_time = 0,
        )
        metrics.save()
        print("-------------------------   Metrics saved to db successfully ")
    except Exception as e:
        print("Error in saving metrics")
        print(e)

    # Save the trained model in the specified directory
    try:
        save_dir = "." 
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            if algo_name=="auto-encoder":
                unique_filename = f"model_{servername}_{metrics_created_at_ms}.h5"
                save_path = os.path.join(save_dir, unique_filename)
                model.save(save_path)

            elif algo_name=="random-forest":
                unique_filename = f"model_{servername}_{metrics_created_at_ms}.pkl"
                save_path = os.path.join(save_dir, unique_filename)
                with open(save_path, 'wb') as file:
                    pickle.dump(model, file) 

            elif algo_name=="svc":
                unique_filename = f"model_{servername}_{metrics_created_at_ms}.pkl"
                save_path = os.path.join(save_dir, unique_filename)
                with open(save_path, 'wb') as file:
                    pickle.dump(model, file) 

            elif algo_name=="isolation-forest":
                unique_filename = f"model_{servername}_{metrics_created_at_ms}.pkl"
                save_path = os.path.join(save_dir, unique_filename)
                with open(save_path, 'wb') as file:
                    pickle.dump(model, file) 
        print(f"-------------------------  Model saved to {save_path} ")

        # Saving Scaler
        if scaler:
            save_dir = "." 
            unique_filename = f"scaler_{servername}_{metrics_created_at_ms}.pkl"
            save_path = os.path.join(save_dir, unique_filename)
            with open(save_path, 'wb') as f:
                pickle.dump(scaler, f)
        print(f"-------------------------  Scaled saved to {save_path} ")

    except Exception as e:
        print(e)
        print("model not saved on the disk")

    return "complete "

"""---------------------------------------  Manual Models Helpers ---------------------------"""


import joblib
import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam



def predict_probs(losses, threshold):
    probabilities = []
    for loss in losses:
        if loss > threshold:
            probabilities.append(loss*100)
        else:
            probabilities.append((1 - loss)*100)
    return np.array(probabilities)


def run_autoencoder_production(input_data, model_period = "Month"):

    if model_period == "Month":
        model_path = "/mnt/ESP_Normal_Models/autoencoder_model_month.h5"
        scaler_path = "/mnt/ESP_Normal_Models/autoencoder_scaler_month.pkl"

    elif model_period == "Week":
        model_path = "/mnt/ESP_Normal_Models/autoencoder_model_week.h5"
        scaler_path = "/mnt/ESP_Normal_Models/autoencoder_scaler_week.pkl"

    elif model_period == "Day":
        model_path = "/mnt/ESP_Normal_Models/autoencoder_model_day.h5"
        scaler_path = "/mnt/ESP_Normal_Models/autoencoder_scaler_day.pkl"

    elif model_period == "6Hours":
        model_path = "/mnt/ESP_Normal_Models/autoencoder_model_6Hours.h5"
        scaler_path = "/mnt/ESP_Normal_Models/autoencoder_scaler_6Hours.pkl"

    else:
        print("Error in model period")

    # Load the autoencoder model with custom optimizer
    custom_objects = {'Adam': Adam}        
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"--------------- Model {model_period} Loaded successfully")
    except Exception as e:
        print("\n Error in Model Loading \n ")
        print(e, "\n")
        return None
    

    # Load the scaler
    scaler = joblib.load(scaler_path)
    print(f"--------------- Scaler {model_period} Loaded successfully")
        

    input_data = input_data.set_index('0')

    input_ = input_data.copy() 

    # Copy the columns and Index
    cols = input_data.columns
    indx = input_data.index

    # Scale the input data
    input_scaled = scaler.fit_transform(input_)
    input_scaled = pd.DataFrame(input_scaled,columns=cols,index = indx)
    reconstructions = model.predict(input_scaled)
    mse_2 = np.array(tf.keras.losses.mae(reconstructions, input_scaled))
    input_data['Probabilities'] = predict_probs(mse_2,95)
    input_data['Probabilities'] = input_data['Probabilities'].clip(lower=0, upper=100)

    output_data = input_data

    return output_data