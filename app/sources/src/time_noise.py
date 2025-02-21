import pandas as pd
import numpy as np
import random
from scipy.interpolate import interp1d






def preprocess(dff):
  dff.drop_duplicates(inplace=True)
  dff = dff.interpolate(method='bfill',limit=50).interpolate(method='pad',limit=50)
  dff.dropna(axis=0,inplace=True)
  return dff


def simulate(y_sample ,extend_to=1440):
    x_sample = np.linspace(0,  len(y_sample), num=len(y_sample))
    f = interp1d(x_sample, y_sample)
    x_sample_new = np.linspace(0,len(y_sample), num=extend_to)
    y_sample_new = f(x_sample_new)
    return y_sample_new




def generate_time_noise_col(df = None, percentage = 0.00):
    all_var = {}

    for j in df.columns:
        # 1 time processing :
        sim = df[j]
        sim_array = sim.values.astype(float)  # Convert values to float and store in a NumPy array
        all_var[j] = sim_array
        # 2 noise processing :
        n = np.random.normal(0, all_var[j].std(), all_var[j].size)*percentage
        corrected_noise = n
        all_var[j] = all_var[j] + corrected_noise
        result_df = pd.DataFrame(all_var)

    return result_df


def generate_time_noise(df=None, samples=1, percentage=0.0):
    noise_percentages = np.random.uniform(0,percentage,samples)
    all_data = []
    for ni in noise_percentages:
        res_i = generate_time_noise_col(df=df, percentage=ni)
        all_data.append(res_i)

    ## concatenate the training splits
    df_all = pd.concat(all_data)

    return df_all



def generate_synthetic_data(df=None, samples=1, percentage=0.1, days=None):
    noise_percentages = np.random.uniform(0,percentage,samples)
    all_data = []
    for ni in noise_percentages:
        res_i = generate_time_noise_col(df=df, percentage=ni)
        all_data.append(res_i)

    return all_data









# def simulate(y_sample ,extend_to=1440):
#     x_sample = np.linspace(0,  len(y_sample), num=len(y_sample))
#     f = interp1d(x_sample, y_sample)
#     x_sample_new = np.linspace(0,len(y_sample), num=extend_to)
#     y_sample_new = f(x_sample_new)
#     return y_sample_new



# def generate_time_noise_col(df = None , days = None, percentage = 0.05):
#     all_var = {}
#     days = days
    
#     for j in df.columns:
#         # 1 time processing :

#         sim = simulate(df[j], extend_to=len(df[j]) if days == 0 else 1440*days)
#         all_var[j] = sim

#         # 2 noise processing :
#         n = np.random.normal(0, all_var[j].std(), all_var[j].size)*percentage
#         corrected_noise = n 
#         all_var[j] = all_var[j] + corrected_noise
        
#         result_df = pd.DataFrame(all_var)
        
#     return result_df


# def generate_time_noise(df=None, samples=1, percentage=0.1, days=None):
#     noise_percentages = np.random.uniform(0,percentage,samples)
#     all_data = []
#     for ni in noise_percentages:
#         res_i = generate_time_noise_col(df=df, days=days, percentage=ni)
#         all_data.append(res_i)

#     ## concatenate the training splits
#     df_all = pd.concat(all_data)

    # return df_all
