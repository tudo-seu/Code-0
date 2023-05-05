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
print(f)


'''
for i in f:
    result = df[df['Measure'] == i]
    result = result[result.index % 4 != 0]      #drop every 4th line
    #sns.relplot(x='time', y='kWh', data=result, kind='line')

    # plt.show()
    t = input()
    
'''

curr = df[df['Measure'] == 'FB3 - Main 3L']

### Generating Test and Training sets
size = len(curr)

test_indices = np.random.choice(size, (size*0.2), replace=False)
all_indices = np.arange(size)
train_indices = np.setdiff1d(all_indices, test_indices)


