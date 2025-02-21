from sources.src.upload_data import load_data
from sources.src.time_noise import generate_time_noise
from sources.src.scale_train_evaluate import train_DL, create_vae
from sources.src.scale_train_evaluate import  threshold_calculation, scaling
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC  # Import Support Vector Classifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def main(servername="intelift", pattern_ranges=None, batch_size=256,
        encoding_dim=64, epochs=30, tags=None):


        ## call the load data method to retrieve pattern data
        dfs = []
        try:
            for time_range in pattern_ranges:
                start, end, well_id = time_range[0], time_range[1], time_range[2]
                df_p = load_data(servername=str(servername), start=int(start), end=int(end), tags=list(tags), wellid=int(well_id))
                dfs.append(df_p)
                print(f"Loaded data for well ID: {well_id}")
            
            df_all = pd.concat(dfs, ignore_index=True)
            print(f"------------------------- Data loaded successfully and has a length of {df_all.shape} -------------------------")
        except Exception as e:
            print("Error in search logic function for data loading")
            print(e)

        # processig Time to Index
        df = df_all.set_index('0')

        # scale training
        scaler = MaxAbsScaler() 
        cols = df.columns
        indx = df.index
        df = scaler.fit_transform(df)
        df = pd.DataFrame(df,columns=cols,index = indx)
    
        # Ensure the data is numeric and has no missing values
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Convert DataFrame to NumPy array
        data = df.to_numpy()

        input_dim = data.shape[1]
        encoding_dim = 64 

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)

        # Compile the model
        autoencoder.compile(optimizer=Adam(), loss='mse', metrics=["accuracy"])

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

        # Train the model
        history = autoencoder.fit(data, data,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=False,
                        validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr]).history
        

        loss, accuracy, val_loss, val_accuracy = history["loss"], history["accuracy"], history["val_loss"], history["val_accuracy"]

        results = {"training_loss":loss[-1], "training_accuracy":accuracy[-1], 
                    "validation_loss":val_loss[-1], "validation_accuracy":val_accuracy[-1],
                    "training_loss_values":loss, "training_accuracy_values":accuracy,
                    "validation_loss_values":val_loss, "validation_accuracy_values":val_accuracy,"model":autoencoder, "scaler":scaler}
        

        print(" ------------------------- training was finished -------------------------")
        

        return results



"""---------------------  this is the main function for random forest algorithm ---------------------------"""



def main_rf(servername="stage", pattern_ranges=None, normal_ranges=None, tags=None):
        

        ## call the load data method to retrieve pattern data
        pattern_data = []
        try:
            for time_range in pattern_ranges:
                start, end, well_id = time_range[0], time_range[1], time_range[2]
                df_p = load_data(servername= str(servername), start=int(start), end= int(end), tags=list(tags), wellid=int(well_id))
                pattern_data.append(df_p)        
                print(f"Loaded data for well ID: {well_id}")


            pattern_data = pd.concat(pattern_data, ignore_index=True)
            pattern_data = pattern_data.set_index('0')
            pattern_data["label"] = len(pattern_data)*[1]
            print(f"------------------------- pattern data is loaded successfully and has a  length  of {pattern_data.shape} --------")
        except Exception as e:
             print("error in search logic function for data normal")
             print(e)


             
        ## call the load data method to retrieve normal data
        normal_data = []
        try:
            for time_range in normal_ranges:
                start, end, well_id = time_range[0], time_range[1], time_range[2]
                df_n = load_data(servername= str(servername), start=int(start), end= int(end), tags=list(tags), wellid=int(well_id))
                normal_data.append(df_n)        
                print(f"Loaded data for well ID: {well_id}")


            normal_data = pd.concat(normal_data, ignore_index=True)
            normal_data = normal_data.set_index('0')
            normal_data["label"] = len(normal_data)*[0]
            print(f"------------------------- normal data is loaded successfully and has a  length  of {normal_data.shape} --------")
        except Exception as e:
             print("error in search logic function for data normal")
             print(e)


        data = pd.concat([normal_data, pattern_data])
        print("------------------------- data is concatenated successfully -------------------------")


        # Scale the selected_data
        # Step 1: Remove the 'failure' column from the dataframe
        # data, scaler = scaling(data)


        X = data.drop("label",axis=1)
        y = data["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
        rf.fit(X_train_scaled,y_train)
        y_pred_test = rf.predict(X_test_scaled)
        print(accuracy_score(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))

        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        confusion = confusion_matrix(y_test, y_pred_test)

        print("Validation accuracy: {:.2f}".format(accuracy))
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))
        print("F1-score: {:.2f}".format(f1))
        print("Confusion Matrix:\n", confusion)

        results = {"training_loss":precision, "training_accuracy":accuracy, 
                    "validation_loss":recall, "validation_accuracy":f1,
                    "training_loss_values":confusion[0][0], "training_accuracy_values":confusion[0][1],
                    "validation_loss_values":confusion[1][0], "validation_accuracy_values":confusion[1][1],
                    "model":rf, "scaler":scaler}
        
        
        return results


"""----------------------------------------------- SVC --------------------------------------------------"""


def main_svc(servername="stage", 
            pattern_ranges=None, 
            normal_ranges=None, 
            tags=None):

        

        ## call the load data method to retrieve pattern data
        pattern_data = []
        try:
            for range in pattern_ranges:
                start, end, wellid = range[0], range[1], range[2]
                df_p = load_data(servername= str(servername), start=int(start), end= int(end), tags=list(tags), wellid=int(wellid))
                pattern_data.append(df_p)        

            pattern_data = pd.concat(pattern_data, ignore_index=True)
            pattern_data = pattern_data.set_index('0')
            pattern_data["label"] = len(pattern_data)*[1]
            print(f"------------------------- pattern data is loaded successfully and has a  length  of {pattern_data.shape} --------")
        except Exception as e:
             print("error in search logic function for data normal")
             print(e)


             
        ## call the load data method to retrieve normal data
        normal_data = []
        try:
            for range in normal_ranges:
                start, end, wellid = range[0], range[1], range[2]
                df_n = load_data(servername= str(servername), start=int(start), end= int(end), tags=list(tags), wellid=int(wellid))
                normal_data.append(df_n)        

            normal_data = pd.concat(normal_data, ignore_index=True)
            normal_data = normal_data.set_index('0')
            normal_data["label"] = len(normal_data)*[0]
            print(f"------------------------- data normal is loaded successfully and has a  length  of {normal_data.shape} --------")
        except Exception as e:
             print("error in search logic function for data normal")
             print(e)


        data = pd.concat([normal_data, pattern_data])
        print("------------------------- data is concatenated successfully -------------------------")


        # Scale the selected_data
        # Step 1: Remove the 'failure' column from the dataframe
        # data, scaler = scaling(data)


        X = data.drop("label",axis=1)
        y = data["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        SVM = SVC(probability=True, random_state=42)
        SVM.fit(X_train_scaled,y_train)

        y_pred_test = SVM.predict(X_test_scaled)
        print(accuracy_score(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))

        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        confusion = confusion_matrix(y_test, y_pred_test)

        print("Validation accuracy: {:.2f}".format(accuracy))
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))
        print("F1-score: {:.2f}".format(f1))
        print("Confusion Matrix:\n", confusion)

        results = {"training_loss":precision, "training_accuracy":accuracy, 
                    "validation_loss":recall, "validation_accuracy":f1,
                    "training_loss_values":confusion[0][0], "training_accuracy_values":confusion[0][1],
                    "validation_loss_values":confusion[1][0], "validation_accuracy_values":confusion[1][1],
                    "model":SVM, "scaler":scaler}
        
        
        return results





def main_if(servername="stage", wellid=None, start=None, end=None, samples = 10, noise=0.01
            ,n_trees = 100, max_samples="auto",contamination = 0.01, tags=None):
     

        main_logs = []

        # call the search logic method to retrieve data
        try:
            df = load_data(servername=servername, start=start, end=end, tags=tags, wellid=wellid)
            print("------- search logic is completed successfully --------")
        except Exception as e:
             print("error in search logic function")
             print(e)

        # processig
        df = df.set_index('0')



        # scale training
        # scaler = preprocessing.MinMaxScaler() 
        # cols = df.columns
        # indx = df.index
        # df = scaler.fit_transform(df)
        # df = pd.DataFrame(df,columns=cols,index = indx)
    

        # call noise and time methods 
        try:
            df_all = generate_time_noise(df=df, samples=samples, percentage=noise)[tags]
            print("------- noise generation is completed successfully --------")
        except Exception as e:
             print("error in generating time noise")
             print(e)

    
        # call the train method
        try:
            clf = IsolationForest(n_estimators=n_trees, max_samples="auto", contamination=0.01, random_state=42, warm_start=True)
            clf.fit(df_all)
        except Exception as e:
             print("error in training the model")
             print(e)

    
        # Calculate training loss and accuracy
        training_loss = 0.0
        training_accuracy = 0.0

        # Calculate validation loss and accuracy
        validation_loss = 0.0
        validation_accuracy = 0.0

        # Collect loss and accuracy values for each epoch
        training_loss_values = [training_loss]
        training_accuracy_values = [training_accuracy]
        validation_loss_values = [validation_loss]
        validation_accuracy_values = [validation_accuracy]
        

    

        results = {"training_loss":training_loss, "training_accuracy":training_accuracy, 
                    "validation_loss":validation_loss, "validation_accuracy":validation_accuracy,
                    "training_loss_values":training_loss_values, "training_accuracy_values":training_accuracy_values,
                    "validation_loss_values":validation_loss_values, "validation_accuracy_values":validation_accuracy_values,
                    "model":clf}
        
        
        print('------training was finished ----------')

        return results
















"""---------------------------------------  Helper Functions ---------------------------"""

def scaling(data):
    # Scaling data    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data.values)
    data_scaledValues = min_max_scaler.transform(data.values)
    scaled_df = pd.DataFrame(data_scaledValues,columns=data.columns)
    scaled_df = scaled_df.set_index(data.index)
    
    return scaled_df, min_max_scaler


def labeling(data,event):
    # Check if event is either "failure" or "normal"
    if event not in ["failure", "normal"]:
        raise ValueError("Invalid value for 'event' parameter. Allowed values are 'failure' or 'normal'.")
    # Labeling
    if event == 'failure' :
        data['event'] = '1'
    else:
        data['event'] = '0'
        
    return data