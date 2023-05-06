import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math

os.chdir("..")
data = 'data/holdout.csv'
vergleich = pd.read_csv(data)

vergleich = vergleich.drop('Measure', axis=1)
vergleich['time'] = pd.to_datetime(vergleich['time'])
#vergleich = vergleich[vergleich['time'].dt.minute == 0]


vergleich.to_csv('vergleich.csv')




df = pd.read_csv(data)
df = df.dropna()
threshold_up = df['kWh'].quantile(0.95)
threshold_down = df['kWh'].quantile(0.05)
df = df[(df['kWh'] > threshold_down) & (df['kWh'] < threshold_up)]
vergleich = vergleich[(vergleich['kWh'] > threshold_down) & (vergleich['kWh'] < threshold_up)]

df = df.drop('Measure', axis = 1)
'''
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
'''



# ****************************************************** #
#   KNeighbors  #
# Create features
# Create features
n = 2 # Number of previous and next values to include in the mean calculation
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
Kneighbor = KMeans(n_clusters=5, random_state=0)
labels = Kneighbor.fit_predict(X)




df = df.drop([f'kWh_prev{n}_mean', f'kWh_next{n}_mean', 'kWh_prevnext_mean'], axis=1)
# Add the cluster labels to the DataFrame
df['cluster'] = labels

#df['cluster'] = df['cluster'].replace({2 : 'Non-production', 0: 'Power-down', 2: 'Power-up', 4: 'Production'})
grouped = df.groupby('cluster')['kWh'].mean()
max_label = grouped.idxmax()
min_label = grouped.idxmin()

clusters = []
for i in grouped:
    clusters.append(i)
clusters.sort()

for i in range(len(clusters)):
    print(np.where(grouped == clusters[i])[0][0])
    clusters[i] = np.where(grouped == clusters[i])[0][0]

print(clusters)
print(len(df['kWh'])*0.88)
percentage =  len(vergleich.loc[vergleich['label'] == 'unclassified', 'label']) / len(vergleich.loc[vergleich['label'] != 'unclassified', 'label'])
print('Unclassified tags %:',percentage)
print('Theoretical accuracy :', 1-percentage)

df.loc[df['cluster'] == clusters[0], 'label'] = 'Non-production'
df.loc[df['cluster'] == clusters[1], 'label'] = 'Non-production'
df.loc[df['cluster'] == clusters[2], 'label'] = 't'
df.loc[df['cluster'] == clusters[3], 'label'] = 'Production'
df.loc[df['cluster'] == clusters[4], 'label'] = 'Production'
'''
df.loc[df['cluster'] == clusters[5], 'label'] = 'Production'
df.loc[df['cluster'] == clusters[6], 'label'] = 'Production'

df.loc[df['cluster'] == clusters[7], 'label'] = 'Production'

df.loc[df['cluster'] == clusters[8], 'label'] = 'Production'
df.loc[df['cluster'] == clusters[9], 'label'] = 'Production'
df.loc[df['cluster'] == clusters[10], 'label'] = 'Production'
df.loc[df['cluster'] == clusters[11], 'label'] = 'Production'
'''
df.iloc[0, df.columns.get_loc('label')] = 'Power-up'
while 't' in df['label'].values:
    df.loc[(df['label'].shift(1) == 'Power-up') & (df['label'].shift(2) == 'Power-up') & (df['label'] == 't'), 'label'] = 'Production'
    df.loc[(df['label'].shift(1) == 'Power-down') & (df['label'].shift(2) == 'Power-down') & (df['label'] == 't'), 'label'] = 'Non-production'
    df.loc[((df['label'].shift(1) == 'Non-production') & (df['label'].shift(-1) == 'Non-production')) & (df['label'] == 't'), 'label'] = 'Non-production'
    df.loc[((df['label'].shift(1) == 'Production') & (df['label'].shift(-1) == 'Production')) & (df['label'] == 't'), 'label'] = 'Production'
    df.loc[((df['label'].shift(1) == 'Non-production') | (df['label'].shift(1) == 'Power-up')) & (df['label'] == 't'), 'label'] = 'Power-up'
    df.loc[((df['label'].shift(1) == 'Production') | (df['label'].shift(1) == 'Power-down')) & (df['label'] == 't'), 'label'] = 'Power-down'

df.to_csv('test.csv')
print('Actual accuracy: ', accuracy_score(vergleich['label'], df['label']))
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
#df_upscaled = df.resample('15T').interpolate()
df_upscaled = df
# Compare with Values above, True if current Value is bigger, False if not
#df_upscaled['Bool'] = df_upscaled['cluster'] > df_upscaled['cluster'].shift(1)
#mask = df_upscaled['Bool'].isna()
#df_upscaled.loc[mask, 'Bool'] = False

mask = df_upscaled['label'].isna()

#df_upscaled.loc[mask, 'label'] = df_upscaled.loc[mask, 'cluster'].astype(int)

#df_upscaled.loc[mask, 'label'] = df_upscaled.loc[mask, 'label'].astype(int).round()
df_upscaled.loc[mask, 'label'] = 't'
df = df_upscaled
while 't' in df['label'].values:
    df.loc[(df['label'] == 't'), 'label'] = df['label'].shift(1)
print(df)

#df.loc[(df['Bool'] != True) & (df['label'] != 'Production') & (df['label'] != 'Non-production'), 'label'] = 'Power-up'
#df.loc[(df['Bool'] == True) & (df['label'] != 'Production') & (df['label'] != 'Non-production'), 'label'] = 'Power-down'

df = df.drop(['cluster'], axis=1)
df.reset_index(inplace=True)
#df = df[df['time'].dt.minute == 0]

#df.to_csv('test.csv')
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
sns.scatterplot(x='time', y='kWh', data=df, hue='label', palette='bright')
#plt.title('KWh vs. Time')
plt.xlabel('Time')
plt.ylabel('kWh')

# Show the plot
plt.show()