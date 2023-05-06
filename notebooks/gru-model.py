from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import  os
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir("..")

data = 'data/train_data.csv'
df = pd.read_csv(data)
curr = df[df['Measure'] == 'FB3 - Main 3L']

# Format curr
size = len(curr)
curr = curr.drop(columns=['Measure', 'label'])
curr = curr.reset_index(drop=True)

# Create Data sets
test_indices = np.random.choice(size, (int(size*0.2)), replace=False)
all_indices = np.arange(size)
train_indices = np.setdiff1d(all_indices, test_indices)

X_train = curr.loc[train_indices, ('')]

### Generating Test and Training sets
size = len(curr)
curr = curr.drop(columns=['Measure', 'label'])
curr = curr.reset_index(drop=True)

model = Sequential()
model.add(GRU(units=32, input_shape=(96, 1)))
model.add(Dense(units=4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])