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

#Convert tweets into features and label 
trueLabel = df['Label'].replace({'Label': {'negative' : 0, 'positive': 1, 'neutral':2}})
Y = trueLabel.values

tweets = df['Tweets']
tweetsMat = tweets.iloc[:]
tweetsMat = tweetsMat.values
# print(tweetsMat.shape)
count = CountVectorizer()
bagOfWords = count.fit_transform(tweetsMat)
X = bagOfWords.toarray().astype(int)
print(X.shape)
print(Y.shape)
#Show feature names (vocabulary)
# feature_names = count.get_feature_names()
# print(feature_names)

# Train classifier
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, Y)
print("training finished")

#Test on itself
acc = clf.score(X, Y)

#Test on all data
test_df = pd.read_csv(dataPath, sep='\t', header=None, names=['ID', 'Label', 'Orig', 'Tweets'], quoting=csv.QUOTE_ALL, encoding='utf8')
test_df = test_df.drop(['ID', 'Tweets'], axis = 1)
test_df = test_df.dropna()

test_df['Orig'] = test_df.Orig.str.replace('\"', '') #remove double quote
test_df['Orig'] = test_df.Orig.str.replace('\'', '') #remove quote
test_df['Orig'] = test_df.Orig.str.replace(':', '') #remove colon, careful of removing emoticons, must use regex!!
test_df['Orig'] = test_df.Orig.str.replace('\,', '') #remove comma
test_df['Orig'] = test_df.Orig.str.replace('\.', '')

testLabel = test_df['Label'].replace({'Label': {'negative' : 0, 'positive': 1, 'neutral':2}})
test_y = testLabel.iloc[:20000].values

test_tweets = test_df['Orig']
testtweetsMat = test_tweets.iloc[:20000]
testtweetsMat = testtweetsMat.values
print(testtweetsMat.shape)
# count_test = CountVectorizer(vocabulary=count.get_feature_names(), )
bagOfWords_test = count.transform(testtweetsMat)
test_x = bagOfWords_test.toarray().astype(int)
# print(bagOfWords_test)

print(test_y.shape, test_x.shape)
test_acc = clf.score(test_x, test_y)
print(acc, test_acc)