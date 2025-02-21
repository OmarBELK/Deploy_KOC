
from corepredict.models import TrainingStatus, TrainingMetrics
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
from lightgbm import LGBMRegressor 
from xgboost import XGBRegressor

def training(**kwargs):

    servername = kwargs['servername']
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

    # extract the kwargs for the auto-encoder
    pattern_ranges = kwargs['pattern_ranges']
    tags = kwargs['tags']
    target_column = kwargs['target_column']
    model_name = kwargs['model_name']
    model_description = kwargs['model_description']
    is_generic = kwargs['is_generic']

    try:
        # Train the auto-encoder model
        response = main_predict(servername=str(servername),
                                pattern_ranges=pattern_ranges,
                                tags=list(tags),
                                target_column=str(target_column))
        

        print("------------------------- Training is complete")
        status = "complete"
        model = response['model']
        scaler = response['scaler']

    except Exception as e:
        print("Error in the main function")
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
            training_status=status_obj,
            model_name=model_name,
            model_description=model_description,
            is_generic=is_generic,
            inputs=str(tags),
            target_column = str(target_column),
            performance = response["performance"],
            mse = response["mse"],
            y_scaler_value = response["y_scaler_value"],
            created_at=metrics_created_at,
            created_at_ms=metrics_created_at_ms,
            creation_source="user",
            last_trained=status_obj.created_at,
            training_time=0,
            execution_time=0,
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
            unique_filename = f"model_{servername}_{metrics_created_at_ms}.pkl"
            with open(unique_filename, 'wb') as file:
                pickle.dump(model, file) 

        print(f"-------------------------  Model saved to {unique_filename} ")
        if scaler:
            save_dir = "." 
            unique_filename = f"scaler_{servername}_{metrics_created_at_ms}.pkl"
            with open(unique_filename, 'wb') as f:
                pickle.dump(scaler, f)

        print(f"-------------------------  Scaler saved to {unique_filename} ")

    except Exception as e:
        print(e)
        print("model not saved on the disk")

    return "complete "
