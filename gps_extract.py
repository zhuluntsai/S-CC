import pandas as pd
from tqdm import tqdm 
import os

directory = 'Formatted Data'

df_list = []
for root, dirs, files in tqdm(os.walk(directory)):
    for filename in files:
        path = os.path.join(root, filename)
        if 'GPS' in path:
            df = pd.read_excel(path)
            p = path.split('/')
            df['subject'] = [ p[1] for _ in range(len(df)) ]
            df['date'] = [ p[3] for _ in range(len(df)) ]
            df['trip'] = [ p[4] for _ in range(len(df)) ]
            df_list.append(df)

pd.concat(df_list).to_csv('test.csv', index=False)
