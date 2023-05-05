import pandas as pd
import numpy as np
import  os

os.chdir("..")

data = 'data/train_data.csv'
df = pd.read_csv(data)

df.sort_values(by = "Measure")

#df.dropna()     #deleting NaN Values

f = df.loc[:, 'Measure'].unique()

for i in f:
    result = df[df['Measure'] == i]
    print(result.head())
    print(len(result))
    result.to_csv(r'D:\Eigene Dokumente\Desktop\Uni\Hackathon\Code-0\data\sep\Measure_' + i + '.csv')
    print("********************")
#print(f.head())
print(len(f))
