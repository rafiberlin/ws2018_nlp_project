from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import os
import csv


def cleanData(df, col):
    """Perform some data cleaning to remove unwanted character/strings
    Removing characters like : does not affect UTF-8 emoticons, but may affect :D, :), :("""

    df[col] = df[col].str.replace('\"', '') #remove double quote
    df[col] = df[col].str.replace('\'', '') #remove quote
    df[col] = df[col].str.replace(':', '') #remove colon, careful of removing emoticons, must use regex!!
    df[col] = df[col].str.replace('\,', '') #remove comma
    df[col] = df[col].str.replace('\.', '') #remove fullstop

    return df


def convertTrain(dataPath):
    """Convert csv files into features and label matrix for training data"""
    df = pd.read_csv(dataPath, sep='\t', header=None, names=['ID', 'Label', 'Orig', 'Tweets'], quoting=csv.QUOTE_ALL, encoding='utf8')
    df = df.drop(['ID', 'Orig'], axis = 1)
    df = df.dropna()
    df = cleanData(df, 'Tweets')

    # Convert tweets into label
    trueLabel = df['Label'].replace({'Label': {'negative' : 0, 'positive': 1, 'neutral':2}})
    Y = trueLabel.values

    # Convert tweets into features
    tweets = df['Tweets']
    tweetsMat = tweets.iloc[:]
    tweetsMat = tweetsMat.values
    count = CountVectorizer()
    bagOfWords = count.fit_transform(tweetsMat)
    X = bagOfWords.toarray()
    X = (X > 1).astype(int)

    # Show feature names (vocabulary)
    # feature_names = count.get_feature_names()
    # print(feature_names)
    return X, Y, count
    # Count to fit vectorizer for test data. Usually can be done in single step, however as vocabulary of training and test data differs, we can use this as temporary placeholder


def convertTest(dataPath, count):
    """Convert csv files into features and label matrix for test data"""
    df = pd.read_csv(dataPath, sep='\t', header=None, names=['ID', 'Label', 'Orig', 'Tweets'], quoting=csv.QUOTE_ALL, encoding='utf8')
    df = df.drop(['ID', 'Tweets'], axis = 1)
    df = df.dropna()
    df = cleanData(df, 'Orig')

    # Convert tweets into label
    testLabel = df['Label'].replace({'Label': {'negative' : 0, 'positive': 1, 'neutral':2}})
    Y = testLabel.iloc[:20000].values

    # Convert tweets into features
    test_tweets = df['Orig']
    testtweetsMat = test_tweets.iloc[:20000]    #Only do for first 20k tweets. Have memory error issue
    testtweetsMat = testtweetsMat.values
    bagOfWords_test = count.transform(testtweetsMat)
    X = bagOfWords_test.toarray()
    X = (X > 1).astype(int)
    return X, Y


if __name__ == "__main__":
    # Read data from textfile, drop tweet ID and original tweets
    path = os.getcwd()
    dataPath = os.path.join(path, 'process_data/clean_data_result_8000.txt')

    # Train classifier
    X, Y, count = convertTrain(dataPath)
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, Y)
    print("Training finished")

    # Test on itself
    accTrain = clf.score(X, Y)

    # Test on 20k tweets
    # Result using terms frequency is 0.9658666023398866 for accTrain and 0.7099 for accTest, the one that is ran use terms presence.
    Xtest, Ytest = convertTest(dataPath, count)
    accTest = clf.score(Xtest, Ytest)
    print(accTrain, accTest)