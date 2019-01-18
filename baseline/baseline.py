from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import csv
# Read data from textfile, drop tweet ID and original tweets
path = os.getcwd()
dataPath = os.path.join(path, 'process_data/clean_data_result_8000.txt')
df = pd.read_csv(dataPath, sep='\t', header=None, names=['ID', 'Label', 'Orig', 'Tweets'], quoting=csv.QUOTE_ALL, encoding='utf8')
df = df.drop(['ID', 'Orig'], axis = 1)
df = df.dropna()
# print(df.iloc[7768]['Tweets'])
# print(df.iloc[784]['Tweets'])
# print(df.iloc[4459]['Tweets'])
# Perform some data cleaning to remove unwanted character/strings
# Removing characters like : does not affect UTF-8 emoticons, but may affect :D, :), :(

df['Tweets'] = df.Tweets.str.replace('\"', '') #remove double quote
df['Tweets'] = df.Tweets.str.replace('\'', '') #remove quote
df['Tweets'] = df.Tweets.str.replace(':', '') #remove colon, careful of removing emoticons, must use regex!!
df['Tweets'] = df.Tweets.str.replace('\,', '') #remove comma
df['Tweets'] = df.Tweets.str.replace('\.', '')


# print(df.iloc[7768]['Tweets'])
# print(df.iloc[784]['Tweets'])
# print(df.iloc[4459]['Tweets'])

#Convert tweets into features and do one-hot encoding for label
trueLabel = pd.get_dummies(df['Label'], prefix=['Label'])
tweets = df['Tweets']
tweetsMat = tweets.iloc[:]
tweetsMat = tweetsMat.values
# print(tweetsMat.shape)
count = CountVectorizer()
bagOfWords = count.fit_transform(tweetsMat)
X = bagOfWords.toarray()
print(X.shape)

#Show feature names (vocabulary)
feature_names = count.get_feature_names()
print(feature_names)