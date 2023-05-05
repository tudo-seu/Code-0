import pandas as pd
import numpy as np

data = 'train_data.csv'
df = pd.read_csv(data)

df.sort_values(by = "Measure")
print(df.head())