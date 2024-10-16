import os
import pandas as pd
import numpy as np
import random
def generate_float_array(start, end, size):
    return [random.uniform(start, end) for _ in range(size)]

def add_impulse_noise(df, noise_percentage):
    """
    Add impulse noise (salt-and-pepper noise) to a specified percentage of data points in the DataFrame.
    :param df: DataFrame
    :param noise_percentage: Percentage of data points to be changed to noise
    :return: DataFrame with noise added
    """
    noisy_df = df.copy()        
    num_samples = len(noisy_df)
    num_noise_points = int((noise_percentage / 100) * num_samples)     
    noise_indices = np.random.choice(num_samples, num_noise_points, replace=False)
    rand_cols= random.sample(noisy_df.columns.tolist(),  random.randint(1, len(noisy_df.columns)))
    
    for column in rand_cols:
        noisy_df.loc[noise_indices, column] = np.random.choice(generate_float_array(df[column].max() * 1.2, df[column].max() * 1.5, num_noise_points), num_noise_points, replace = False)

    noisy_df['labels'] = 1
    noisy_df.loc[noise_indices, 'labels'] = -1
    return noisy_df

def combine(data_dir):
    dataframes = {}  # Dictionary to store the dataframes with file name as the key

    for file in os.listdir(data_dir):    
        if file.endswith('.csv'):  # Ensure we only process CSV files
            base_name = os.path.splitext(file)[0]  # Extract base name without extension
            base_name = file[4:-4]        
            df_powerMeter = pd.read_csv(os.path.join(data_dir, file))  # Read CSV file into dataframe
            
            if base_name not in dataframes:
                dataframes[base_name] = df_powerMeter
            else:
                dataframes[base_name] = pd.concat([dataframes[base_name], df_powerMeter])
            
    # Save the combined dataframe to CSV
    for base_name, dataframe in dataframes.items():
        dataframe.to_csv(f'combined18/{base_name}.csv', index=False)
                
def fetch_dataset(data_dir):
    dataframes = {}  # Dictionary to store the dataframes with file name as the key
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):  # Ensure we only process CSV files
        
            base_name = file[:-4]        
            print(base_name)
            clean_df = pd.read_csv(os.path.join(data_dir, file))
            clean_df['Date'] = pd.to_datetime(clean_df['Date'], errors= 'coerce')
            clean_df = clean_df.dropna(subset=['Date'])
            
            # Extract year, month, day, hour, and minute components
            clean_df['Date'] = pd.to_datetime(clean_df['Date'], errors= 'coerce')
            clean_df['Year'] = clean_df['Date'].dt.year
            clean_df['Month'] = clean_df['Date'].dt.month
            clean_df['Day'] = clean_df['Date'].dt.day
            clean_df['Hour'] = clean_df['Date'].dt.hour
            clean_df['Minute'] = clean_df['Date'].dt.minute
        
            # set date as index column
            clean_df = clean_df.set_index('Date')
            # Leave columns with keyword of 'kW' and calculate sum of energy consumption of all floors 
            tmp1 = clean_df.loc[:, clean_df.columns.str.contains('kW')].groupby('Date').sum()
            # sum of energy consumption of all power outlets in all floors and zones
            # tmp2 = tmp1.sum(axis=1).rename('total_demand').to_frame()
            
            # df_powerMeter = pd.concat([tmp2, tmp1], axis=1)
            tmp2 = tmp1.sum(axis=1).rename('total_demand').to_frame()
            tmp3 = clean_df.loc[:, clean_df.columns.str.contains('AC')].groupby('Date').sum().sum(axis=1).rename('AC').to_frame()
            tmp4 = clean_df.loc[:, clean_df.columns.str.contains('Light')].groupby('Date').sum().sum(axis=1).rename('Light').to_frame()
            tmp5 = clean_df.loc[:, clean_df.columns.str.contains('Plug')].groupby('Date').sum().sum(axis=1).rename('Plug').to_frame()

            df_powerMeter = pd.concat([tmp2, tmp1, tmp3,tmp4,tmp5], axis=1)

            # mean of each T mins
            df_powerMeter = df_powerMeter.resample('10min').mean()
    
            # Extract the date components
            df_powerMeter['Year'] =  pd.DatetimeIndex(df_powerMeter.index).year
            df_powerMeter['Month'] = pd.DatetimeIndex(df_powerMeter.index).month
            df_powerMeter['Day'] = pd.DatetimeIndex(df_powerMeter.index).day
            df_powerMeter['Hour'] = pd.DatetimeIndex(df_powerMeter.index).hour
            df_powerMeter['Minute'] = pd.DatetimeIndex(df_powerMeter.index).minute
    
            df_powerMeter.index = pd.to_datetime(df_powerMeter.index, errors='coerce')
            # df_powerMeter = df_powerMeter.dropna(subset = ['z1_AC1(kW)', 'z1_AC2(kW)', 'z1_AC3(kW)', 'z1_AC4(kW)', 'z1_Light(kW)', 'z1_Plug(kW)', 'z2_AC1(kW)', 'z2_Light(kW)', 'z2_Plug(kW)', 'z3_Light(kW)', 'z3_Plug(kW)','z4_AC1(kW)', 'z4_Light(kW)', 'z4_Plug(kW)', 'z5_AC1(kW)', 'z5_Light(kW)', 'z5_Plug(kW)'], how='all')
            df_powerMeter = df_powerMeter[['total_demand','AC', 'Light', 'Plug', 'Year', 'Month', 'Day', 'Hour', 'Minute']]
            df_powerMeter = df_powerMeter.dropna(how='any')
            
            df_powerMeter.sort_values('Date')
            
            dataframes[base_name] = df_powerMeter
            
    return dataframes   