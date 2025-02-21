import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.regularizers as regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import keras.backend as K
from sklearn.model_selection import train_test_split



def scaling(data):
    # Scaling data    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data.values)
    data_scaledValues = min_max_scaler.transform(data.values)
    scaled_df = pd.DataFrame(data_scaledValues,columns=data.columns)
    scaled_df = scaled_df.set_index(data.index)
    
    return scaled_df, min_max_scaler


RANDOM_SEED=27  

def split_data(df=None):

    logs = []

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, shuffle=False)

    logs.append(f"X_train was scaled and has a shape of {X_train.shape}")
    logs.append(f"X_test was scaled and has a shape of {X_test.shape}")


    ## create a helper vector Y
    y_test = np.ones(len(X_test))

    return X_train, X_test, logs

def train_DL(X_train=None, X_test=None, batch_size=100, latent_dim=10, intermediate_dim1=25,
             intermediate_dim2=15, epochs=5, epsilon_std=1.0):

    ## Defining the key parameters
    batch_size = batch_size
    original_dim = X_train.shape[1]
    latent_dim = latent_dim
    intermediate_dim1 = intermediate_dim1
    intermediate_dim2 = intermediate_dim2
    epochs = epochs
    epsilon_std = epsilon_std

    ###########################
    # Input to our encoder
    ###########################
    x = Input(shape=(original_dim,), name="input")

    # intermediate layer
    h1 = Dense(intermediate_dim1, activation="tanh", name="encoding1", 
            activity_regularizer=regularizers.l1(10e-5))(x)

    h  = Dense(intermediate_dim2, activation="relu", name="encoding")(h1)

    # defining the mean of the latent space.
    z = Dense(latent_dim, activation = "relu", name="encoder_out")(h)

    # defining the encoder as a keras model
    encoder = Model(x, z, name="encoder")

    # print out summary of what we just did 
    encoder.summary()

    # Input to our decoder

    input_decoder = Input(shape=(latent_dim,), name="decoder_input")
    # taking the latent space to intermediate dimension
    decoder_h1 = Dense(intermediate_dim2, activation="relu", name="decoder_h1")(input_decoder)
    decoder_h = Dense(intermediate_dim1, activation="relu", name="decoder_h")(decoder_h1)

    # getting the mean from the original dimension
    x_decoded = Dense(original_dim, activation="tanh", name="flat_decoded")(decoder_h)

    # defining the decoder as a keras model
    decoder = Model(input_decoder, x_decoded, name="decoder")

    decoder.summary()

    # AutoEncoder

    # grab the output. Recall, that we need to grab the 3rd element our sampling z
    output_combined = decoder(encoder(x))
    # link the input and the overall output.
    autoencoder = Model(x, output_combined)

    # print out what the overall model looks like
    autoencoder.summary()

    # train the neural network  
    autoencoder.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

    # history = autoencoder.fit(X_train, X_train, 
    #                         epochs=epochs, 
    #                         batch_size = batch_size, 
    #                         validation_data=(X_test, X_test), 
    #                         shuffle=True,
    #                         verbose=1).history

    history = autoencoder.fit(X_train, X_train, 
                             epochs=epochs, 
                             batch_size = batch_size, 
                             validation_data=(X_test, X_test), 
                             shuffle=True,
                             verbose=1).history

    loss, accuracy, val_loss, val_accuracy = history["loss"], history["accuracy"], history["val_loss"], history["val_accuracy"]


    #evaluation = autoencoder.evaluate(X_test, X_test)
    return autoencoder ,loss, accuracy, val_loss, val_accuracy




def create_vae(X_train, X_test, original_dim, intermediate_dim=16, latent_dim=32, epochs=5, batch_size=32):
    inputs = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    vae = Model(inputs, x_decoded_mean)

    reconstruction_loss = tf.keras.losses.mse(inputs, x_decoded_mean)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam', metrics=["accuracy"])

    history = vae.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, X_test))
    
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["accuracy"]

    
    return vae ,loss, accuracy, val_loss, val_accuracy



def evaluate(model, X_train, X_test):
    # calculate the threshold
    reconstructions = model.predict(X_train)
    train_loss = tf.keras.losses.mae(reconstructions, X_train)
    threshold = np.mean(train_loss) + np.std(train_loss)

    # calculate predictions [True, False, ..]
    predictions = predict(model=model, data=X_test, threshold=threshold)
    Accuracy, Precision, Recall = calculate_stats(predictions=predictions, labels= np.ones(len(X_test)))

    return Accuracy, Precision, Recall, threshold

def threshold_calculation(model, X_train):
    reconstructions = model.predict(X_train)
    train_loss = tf.keras.losses.mae(reconstructions, X_train)
    threshold = np.mean(train_loss) + np.std(train_loss)

    return threshold

def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def calculate_stats(predictions, labels):
    Accuracy = accuracy_score(labels, predictions)
    Precision = precision_score(labels, predictions)
    Recall = recall_score(labels, predictions)
    return Accuracy, Precision, Recall
