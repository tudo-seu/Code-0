from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir("..")

data = 'data/train_data.csv'
df = pd.read_csv(data)
fac_loc_list = df.loc[:, 'Measure'].unique()

for fac_loc in fac_loc_list:
    result = df[df['Measure'] == fac_loc]


    if 'Production' in result['label'].unique():
        #result = result.loc[result['label'] != 'unclassified']
        print('Current Measure: ' + fac_loc)
        print(result['label'].value_counts())
        print(result.describe())
        print(result.info())
        print(result.isna().sum())

        result['time'] = pd.to_datetime(result['time'])




        result = result[result['time'].dt.minute == 0]

        print(result.head())
        result = result.set_index('time')

        # drop outliers
        threshold_up = result['kWh'].quantile(0.95)
        threshold_down = result['kWh'].quantile(0.05)
        result = result[(result['kWh'] < threshold_up) & (result['kWh'] > threshold_down)]

        result.to_csv(fac_loc + '.csv')

        #g = sns.relplot(x=result.index, y='kWh', data=result, kind='line', hue='label')
        #plt.xticks(rotation=45)
        #plt.show()