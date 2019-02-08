import os
import sys
sys.path.insert(0, os.getcwd())
import math
from process_data.helper import get_tagged_sentences, get_labels, extract_range
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from baseline.baseline import do_not_tokenize
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix


def calculate_ocfs_score(fitted_docs, labels):
    """
    Calculate for each feature the OCFS Score, returned in a vector
    :param fitted_docs: Matrix as returned by fit_transform() in sklearn
    :param labels: Matrix as returned by extract_range() when used to extract the labels
    :return: a vector of OCFS scores
    """
    # fitted_docs_pd_frame = pd.DataFrame(fitted_docs.toarray())
    # fitted_docs_pd_frame["Label"] = labels["Label"]
    # print(fitted_docs_pd_frame)
    # totalSamples = len(fitted_docs_pd_frame)
    # all_mean = fitted_docs_pd_frame.mean(axis=0)
    # print(all_mean)
    # class_mean = fitted_docs_pd_frame.groupby("Label").mean()
    # print(class_mean)
    # classSamples = fitted_docs_pd_frame.groupby("Label").size()
    # # Sorry for that one, I tried to keep it compact due to memory issues
    # # The argument of np.sum is a list of the mean of features for each class
    # # when the mean of a featur is calculated for a class, it gets added along the rows (axis=0)
    # # to get a feature score for each feature...
    # ocfs = np.sum([(classSamples[label]/totalSamples)*np.square(class_mean.loc[label,] - all_mean) for label in class_mean.index], axis=0)
    # # print(np.square(class_mean.iloc[idx,] - all_mean))
    # # print(class_mean.loc[class_mean['Label'] == "positive"])
    docs = pd.SparseDataFrame(fitted_docs)
    docs = docs.fillna(0) 
    docs["Label"] = labels["Label"]
    totalSamples = len(docs)
    allMean = fitted_docs.mean(axis=0)   #  Don't know why mean with dataframe directly yields inaccurate results
    allMean = pd.Series(np.ravel(allMean))
    classMean = docs.groupby("Label").mean()
    classSamples = docs.groupby("Label").size()
    ocfs = np.sum([(classSamples[label]/totalSamples)*np.square(classMean.loc[label] - allMean) for label in classMean.index], axis=0)
    return ocfs


def retrieve_features_to_remove(ocfs, lowest_val, highest_val):
    """
    Return the index of the features to discard, based on some boundary values
    :param ocfs:
    :param lowest_val:
    :param highest_val:
    :return:
    """

    return [idx for idx, val in enumerate(ocfs) if val < lowest_val or val > highest_val]


def gen_pos_features(docs, tags, weight):
    """
    Generate POS features for given docs and tags based on specific weighting scheme.
    :param docs:
    :param tags:
    :param weight:
    :return:
    """
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for d, e in zip(docs, tags):
        for term, pos in zip(d, e):
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            val = weight.setdefault(pos, 0)
            data.append(val)
        indptr.append(len(indices))
    pos_train = csr_matrix((data, indices, indptr), dtype=float)
    pos_train_normalized = normalize(pos_train, norm='l1', copy=False)
    return pos_train_normalized


def drop_cols(matrix, drop_idx):
    """
    Drop column given index to be dropped.Based on https://stackoverflow.com/questions/23966923/delete-columns-of-matrix-of-csr-format-in-python
    :param matrix: 
    :param drop_idx: 
    :return:
    """
    drop_idx = np.unique(drop_idx)
    tempMat = matrix.tocoo()
    keep = ~np.in1d(tempMat.col, drop_idx)
    tempMat.data, tempMat.row, tempMat.col = tempMat.data[keep], tempMat.row[keep], tempMat.col[keep]
    tempMat.col -= drop_idx.searchsorted(tempMat.col)  #  decrease column indices
    tempMat._shape = (tempMat.shape[0], tempMat.shape[1] - len(drop_idx))
    return tempMat.tocsr()


def main():
    # parent_dir = Path(__file__).parents[1]
    parent_dir = os.getcwd() # my sys.path is different from PyCharm
    DATA_SET_PATH = os.path.join(parent_dir, "dataset/raw_data_by_year/train/")
    TAGGED_SENTENCES = os.path.join(DATA_SET_PATH, 'text_cleaned_pos.csv')
    LABELS = os.path.join(DATA_SET_PATH, 'shuffled.csv')

    all_docs, all_tags = get_tagged_sentences(DATA_SET_PATH, TAGGED_SENTENCES)
    all_labels = get_labels(LABELS)
    samplesLen = len(all_labels)
    testEnd = math.floor(0.3 * samplesLen)  # 20% for test
    devEnd = math.floor(0.1 * samplesLen)  # 10% for dev
    
    DEV_RANGE = (0, devEnd)
    TEST_RANGE = (devEnd, testEnd)
    TRAINING_RANGE = (testEnd, samplesLen)

    dev_docs = extract_range(all_docs, DEV_RANGE[0], DEV_RANGE[1])
    dev_labels = extract_range(all_labels, DEV_RANGE[0], DEV_RANGE[1])
    dev_tags = extract_range(all_tags, DEV_RANGE[0], DEV_RANGE[1])
    # work around to prevent scikit performing tokenizing on already tokenized documents...
    bag_of_words = CountVectorizer(
        analyzer='word',
        tokenizer=do_not_tokenize,
        preprocessor=do_not_tokenize,
        token_pattern=None,
        # stop_words="english"
        binary=True  # replaces bow_train = (bow_train >= 1).astype(int)
    )
    bow_train = bag_of_words.fit_transform(all_docs)
    ocfs = calculate_ocfs_score(bow_train, all_labels)
    feature_idx = retrieve_features_to_remove(ocfs, 10 ** -7, 10 ** -2)
    print(bow_train.shape)
    # print(feature_idx)
    # print(len(feature_idx))
    # vocabulary = dict(map(reversed, bag_of_words.vocabulary_.items()))
    # words_to_ignore = [vocabulary[idx] for idx in feature_idx]

    dev_labels = np.ravel(dev_labels)
    all_labels = np.ravel(all_labels)

    # Removing the features by setting them to zero...
    # pd_bow_train = pd.DataFrame(bow_train.toarray())
    # for idx in feature_idx:
    #     pd_bow_train.iloc[:, idx] = 0
    pd_bow_train = drop_cols(bow_train, feature_idx)
    print(pd_bow_train.shape)

    # Train classifier on Bag of Words (Term Presence) and TF-IDF
    bow_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                        multi_class='multinomial',
                                        max_iter=5000
                                        ).fit(pd_bow_train, all_labels)
    bow_test_acc = bow_classifier.score(pd_bow_train, all_labels)

    print(bow_test_acc)

    # pos_vocab = {'N': 2, 'V':3, 'A':4, 'R':5}  # 5 for N, 3 for V, 2 for A, 1 for R
    # pos_train = gen_pos_features(all_docs, all_tags, pos_vocab)
    # pos_classifier = LogisticRegression(random_state=0, solver='lbfgs',
    #                                     multi_class='multinomial',
    #                                     max_iter=5000
    #                                     ).fit(pos_train, all_labels)
    # pos_train_acc = pos_classifier.score(pos_train, all_labels)
    # print(pos_train_acc)
    
if __name__ == "__main__":
    main()
