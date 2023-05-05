import pandas as pd
import numpy as np
import  os
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("..")

data = 'data/train_data.csv'
df = pd.read_csv(data)

df.sort_values(by = "Measure")

#df.dropna()     #deleting NaN Values

f = df.loc[:, 'Measure'].unique()


for i in f:
    result = df[df['Measure'] == i]
    result = result[result.index % 4 != 0]
    sns.relplot(x='time', y='kWh', data=result, kind='line')

    plt.show()
    t = input()
print(len(f))
