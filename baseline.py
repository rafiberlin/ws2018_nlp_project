from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import nltk.corpus.reader.conll as conll
from process_data.helper import get_labels, get_tagged_sentences, extract_range
import pandas as pd
import numpy as np
import os
import csv
import math


# def cleanData(df, col): # will be removed later
#     """Perform some data cleaning to remove unwanted character/strings
#     Removing characters like : does not affect UTF-8 emoticons, but may affect :D, :), :("""

#     df[col] = df[col].str.replace('\"', '') #remove double quote
#     df[col] = df[col].str.replace('\'', '') #remove quote
#     df[col] = df[col].str.replace(':', '') #remove colon, careful of removing emoticons, must use regex!!
#     df[col] = df[col].str.replace('\,', '') #remove comma
#     df[col] = df[col].str.replace('\.', '') #remove fullstop

#     return df


# def convertTrain(dataPath): # will be removed later
#     """Convert csv files into features and label matrix for training data"""
#     df = pd.read_csv(dataPath, sep='\t', header=None, names=['ID', 'Label', 'Orig', 'Tweets'], quoting=csv.QUOTE_ALL, encoding='utf8')
#     df = df.drop(['ID', 'Orig'], axis = 1)
#     df = df.dropna()
#     df = cleanData(df, 'Tweets')

#     # Convert tweets into label
#     trueLabel = df['Label'].replace({'Label': {'negative' : 0, 'positive': 1, 'neutral':2}})
#     Y = trueLabel.values

#     # Convert tweets into features
#     tweets = df['Tweets']
#     tweetsMat = tweets.iloc[:]
#     tweetsMat = tweetsMat.values
#     count = CountVectorizer()
#     bagOfWords = count.fit_transform(tweetsMat)
#     X = bagOfWords.toarray()
#     X = (X > 1).astype(int)

#     # Show feature names (vocabulary)
#     # feature_names = count.get_feature_names()
#     # print(feature_names)
#     return X, Y, count
#     # Count to fit vectorizer for test data. Usually can be done in single step, however as vocabulary of training and test data differs, we can use this as temporary placeholder


# def convertTest(dataPath, count): # will be removed later
#     """Convert csv files into features and label matrix for test data"""
#     df = pd.read_csv(dataPath, sep='\t', header=None, names=['ID', 'Label', 'Orig', 'Tweets'], quoting=csv.QUOTE_ALL, encoding='utf8')
#     df = df.drop(['ID', 'Tweets'], axis = 1)
#     df = df.dropna()
#     df = cleanData(df, 'Orig')

#     # Convert tweets into label
#     testLabel = df['Label'].replace({'Label': {'negative' : 0, 'positive': 1, 'neutral':2}})
#     Y = testLabel.iloc[:20000].values

#     # Convert tweets into features
#     test_tweets = df['Orig']
#     testtweetsMat = test_tweets.iloc[:20000]    #Only do for first 20k tweets. Have memory error issue
#     testtweetsMat = testtweetsMat.values
#     bagOfWords_test = count.transform(testtweetsMat)
#     X = bagOfWords_test.toarray()
#     X = (X > 1).astype(int)
#     return X, Y


def do_not_tokenize(doc):
    return doc


def main():
    # Do we still need punctuation removal? at least it can reduce feature space seeing that although
    # the tokenization is good there is still to many useless punctuation.
    MAIN_FOLDER = "dataset"
    path = os.path.join(os.getcwd(), MAIN_FOLDER)
    TAGGED_SENTENCES = os.path.join(path, 'text_cleaned_pos.csv')
    LABELS = os.path.join(path, 'shuffled.csv')
    docs, tags = get_tagged_sentences(MAIN_FOLDER, TAGGED_SENTENCES, start_range= 1)
    labels = get_labels(LABELS, start_range=1)
    dataLen = len(labels)
    trainEnd = math.floor(0.7 * dataLen) # 70% for train
    testStart = math.floor(0.8 * dataLen) # 20% for test
    train_docs, test_docs = docs[:trainEnd], docs[testStart:]
    train_labels, test_labels = labels[:trainEnd], labels[testStart:]

    common_stop_words = set(stopwords.words('english'))

    # work around to prevent scikit performing tokenizing on already tokenized documents...
    bag_of_words = CountVectorizer(
        analyzer='word',
        tokenizer=do_not_tokenize,
        preprocessor=do_not_tokenize,
        token_pattern=None,
        #stop_words="english"
    )

    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=do_not_tokenize,
        preprocessor=do_not_tokenize,
        token_pattern=None,
        #stop_words="english"
    )

    # Convert data into features for training
    bow_train = bag_of_words.fit_transform(train_docs)  # matrix is of type CSR
    bow_train = (bow_train >= 1).astype(int)
    tfidf_train = tfidf.fit_transform(train_docs)
    train_labels = np.ravel(train_labels)  # return flat array of labels

    # Train classifier on Bag of Words (Term Presence) and TF-IDF
    bow_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                        multi_class='multinomial',
                                        max_iter=5000
                                        ).fit(bow_train, train_labels)
    tf_idf_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                           multi_class='multinomial',
                                           max_iter=5000
                                           ).fit(tfidf_train, train_labels)

    # Test on itself for BoW and TF-IDF
    bow_train_acc = bow_classifier.score(bow_train, train_labels)
    tfidf_train_acc = tf_idf_classifier.score(tfidf_train, train_labels)
    print("Training score BOW", bow_train_acc)
    print("Training score TFIDF", tfidf_train_acc)

    # Convert data into features for testing
    bow_test = bag_of_words.transform(test_docs)
    bow_test = (bow_test >= 1).astype(int)
    tfidf_test = tfidf.transform(test_docs)
    test_labels = np.ravel(test_labels)

    # Test on test data
    bow_test_acc = bow_classifier.score(bow_test, test_labels)
    tfidf_test_acc = tf_idf_classifier.score(tfidf_test, test_labels)
    print("Testing score BOW", bow_test_acc)
    print("Training score TFIDF", tfidf_test_acc)


if __name__ == "__main__":
    main()
