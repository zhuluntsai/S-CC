import pandas as pd
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt

directory = 'Formatted Data'

df_list = []
speed_list = []
for root, dirs, files in tqdm(os.walk(directory)):
    for filename in files:
        path = os.path.join(root, filename)
        if 'GPS' in path:
            df = pd.read_excel(path)
            p = path.split('/')
            df['subject'] = [ p[1] for _ in range(len(df)) ]
            df['date'] = [ p[3] for _ in range(len(df)) ]
            df['trip'] = [ p[4] for _ in range(len(df)) ]

            max_speed = df['speed'].max()
            if max_speed < 2.5:
                df_list.append(df)
                speed_list.append(max_speed)

plt.hist(speed_list)
plt.savefig('test.png')
# print(speed_list)
pd.concat(df_list).to_csv('test.csv', index=False)
