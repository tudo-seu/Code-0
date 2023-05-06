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
curr = df[df['Measure'] == 'Sub - Feeder F02']
label_map = {'Non-production': 0, 'Power-down': 1, 'Power-up': 2, 'Production': 3}

# Format curr
size = len(curr)
print(curr.head())
curr = curr.dropna()
curr = curr.drop(columns=['Measure'])
curr = curr.reset_index(drop=True)


curr['map'] = curr['label'].map(label_map)

# Create Data sets
test_indices = np.random.choice(size, (int(size*0.2)), replace=False)
all_indices = np.arange(size)
train_indices = np.setdiff1d(all_indices, test_indices)
train = curr.iloc[:-(curr.shape[0] % 96)]



x_train = train.loc[train_indices]['kWh'].values.reshape(-1, 96, 1)
y_train = train.loc[train_indices]['map'].values.reshape(-1, 1)

x_val = curr.loc[test_indices, ('kWh')].values.reshape(-1, 96, 1)
y_val = curr.loc[test_indices, ('map')].values.reshape(-1, 1)

print(x_train)
### Generating Test and Training sets



model = Sequential()
model.add(GRU(units=32, input_shape=(96, 1)))
model.add(Dense(units=4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])
