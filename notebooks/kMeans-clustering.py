import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math

os.chdir("..")
data = 'data/supervised/Sub - Feeder F02.csv'
vergleich = pd.read_csv(data)

vergleich = vergleich.drop('Measure', axis=1)
vergleich.to_csv('vergleich.csv')


df = pd.read_csv(data)
df = df.dropna()
threshold_down = df['kWh'].quantile(0.01)
df = df[(df['kWh'] > threshold_down)]

df['num_time'] = pd.to_numeric(pd.to_datetime(df['time']))
df = df.drop(['label', 'Measure', 'time'], axis=1)

kWh = df['kWh'].values.reshape(-1, 1)
time = df['num_time'].values.reshape(-1, 1)

X = np.concatenate((kWh, time), axis=1)
#X = kWh
# Initialize K-Means clustering algorithm
kmeans = KMeans(n_clusters=3)

# Fit the K-Means model to the data
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_
labels = kmeans.predict(X)

# Add the cluster labels as a new column in the original data
df['phase'] = labels

# Save the updated data to a new CSV file
#df.to_csv('Sub - Feeder F01-labeled.csv')
df['time'] = pd.to_datetime(df['num_time'])
df = df.set_index('time')
print(df.head())
sns.set_palette("bright")
#sns.lineplot(x=df.index, y='kWh', hue='phase', data=df)
# Set the plot title and axis labels
#plt.title('KWh vs. Time')
plt.xlabel('Time')
plt.ylabel('kWh')

# Show the plot
#plt.show()




# ****************************************************** #
#   KNeighbors  #
# Create features
# Create features
n = 50 # Number of previous and next values to include in the mean calculation
df[f'kWh_prev{n}_mean'] = df['kWh'].rolling(window=2*n+1, min_periods=1).apply(lambda x: x[:n].mean())
df[f'kWh_next{n}_mean'] = df['kWh'].rolling(window=2*n+1, min_periods=1).apply(lambda x: x[-n:].mean())
df['kWh_prevnext_mean'] = (df[f'kWh_prev{n}_mean'] + df[f'kWh_next{n}_mean']) / 2
X = df[['kWh', f'kWh_prev{n}_mean', f'kWh_next{n}_mean', 'kWh_prevnext_mean']].values
#
#X = df[['kWh', 'kWh_prev3_mean', 'kWh_next3_mean', 'kWh_prevnext3_std']].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.nan_to_num(X, nan=0)
# Apply K-means clustering
Kneighbor = KMeans(n_clusters=4, random_state=0)
labels = Kneighbor.fit_predict(X)




df = df.drop([f'kWh_prev{n}_mean', f'kWh_next{n}_mean', 'kWh_prevnext_mean'], axis=1)
# Add the cluster labels to the DataFrame
df['cluster'] = labels

#df['cluster'] = df['cluster'].replace({2 : 'Non-production', 0: 'Power-down', 2: 'Power-up', 4: 'Production'})
grouped = df.groupby('cluster')['kWh'].mean()
max_label = grouped.idxmax()
min_label = grouped.idxmin()

known = [max_label, min_label]
missing = []
for i in range(4):
    if i not in known:
        missing.append(i)

df.loc[df['cluster'] == min_label, 'label'] = 'Non-production'
df.loc[df['cluster'] == max_label, 'label'] = 'Production'
df.loc[df['cluster'] == missing[0], 'label'] = 'Power-up'
df.loc[df['cluster'] == missing[1], 'label'] = 'Power-down'
df_upscaled = df.resample('15T').interpolate()

mask = df_upscaled['label'].isna()

df_upscaled.loc[mask, 'label'] = df_upscaled.loc[mask, 'cluster'].astype(int)
print(df_upscaled)
#df_upscaled.loc[mask, 'label'] = df_upscaled.loc[mask, 'label'].astype(int).round()
df = df_upscaled
df.loc[df['label'] == min_label, 'label'] = 'Non-production'
df.loc[df['label'] == max_label, 'label'] = 'Production'
df.loc[df['label'] == missing[0], 'label'] = 'Power-up'
df.loc[df['label'] == missing[1], 'label'] = 'Power-down'


df = df.drop(['num_time', 'phase', 'cluster'], axis=1)
df.reset_index(inplace=True)
print(df)
df = df[df['time'].dt.minute == 0]
df.to_csv('test.csv')

'''
for index, row in df.iterrows():
    if math.isnan(row['cluster']):
        if (row-1)['cluster'] != (row+3)['cluster']:
            if (row-1)['cluster'] == 'Production':
                state = 'Power-down'
            else:
                state = 'Power-up'
            for i in range(4):
                (row + i)['cluster'] = state

        else:
            for i in range(4):
                (row + i)['cluster'] = (row-1)['cluster']
'''

sns.set_palette("bright")
#sns.histplot(data=df, x='time', kde=True, element="step")
sns.lineplot(x='time', y='kWh', data=df, hue='label', palette='bright')
#plt.title('KWh vs. Time')
plt.xlabel('Time')
plt.ylabel('kWh')

# Show the plot
plt.show()