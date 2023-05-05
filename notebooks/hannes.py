import pandas as pd
import numpy as np
import  os

os.chdir("..")

data = 'data/train_data.csv'
df = pd.read_csv(data)

df.sort_values(by = "Measure")

df.dropna()     #deleting NaN Values

print(df.head())
