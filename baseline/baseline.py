from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import csv
#Read data from textfile, drop tweet ID and original tweets
path = os.getcwd()
dataPath = os.path.join(path, 'process_data/clean_data_result_8000.txt')
df = pd.read_csv(dataPath, sep='\t', header=None, names=['ID', 'Label', 'Orig', 'Tweets'], quoting=csv.QUOTE_ALL)
df = df.drop(['ID', 'Orig'], axis = 1)
print(df.iloc[8290])

#Perform some data cleaning to remove unwanted character/strings