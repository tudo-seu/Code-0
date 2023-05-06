import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("..")
data = 'data/unsupervised/Sub-FeederF01.csv'


df = pd.read_csv(data)
df = df.dropna()
threshold_down = df['kWh'].quantile(0.07)
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
df.to_csv('Sub - Feeder F01-labeled.csv')
df['time'] = pd.to_datetime(df['num_time'])
df = df.set_index('time')
print(df.head())
sns.set_palette("bright")
sns.lineplot(x=df.index, y='kWh', hue='phase', data=df)
# Set the plot title and axis labels
#plt.title('KWh vs. Time')
plt.xlabel('Time')
plt.ylabel('kWh')

# Show the plot
plt.show()




# ****************************************************** #
#   KNeighbors  #
# Create features
df['kWh_prev3_mean'] = df['kWh'].rolling(window=4, min_periods=1).apply(lambda x: x[0:3].mean())
df['kWh_next3_mean'] = df['kWh'].rolling(window=4, min_periods=1).apply(lambda x: x[1:4].mean())
df['kWh_prevnext3_std'] = df['kWh'].rolling(window=4, min_periods=1).apply(lambda x: x[0:3].std() + x[1:4].std())
X = df[['kWh', 'kWh_prev3_mean', 'kWh_next3_mean', 'kWh_prevnext3_std']].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.nan_to_num()
# Apply K-means clustering
Kneighbor = KMeans(n_clusters=4, random_state=0)
labels = Kneighbor.fit_predict(X)

# Add the cluster labels to the DataFrame
df['cluster'] = labels

print(df.head())
sns.set_palette("bright")
sns.lineplot(x=df.index, y='kWh', hue='cluster', data=df)
# Set the plot title and axis labels
#plt.title('KWh vs. Time')
plt.xlabel('Time')
plt.ylabel('kWh')

# Show the plot
plt.show()