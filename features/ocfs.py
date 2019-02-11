import os
import sys

sys.path.insert(0, os.getcwd())
import math
from process_data.helper import get_tagged_sentences, get_labels, extract_range, pre_processing
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from baseline.baseline import do_not_tokenize
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from collections import defaultdict
import time


def calculate_ocfs_score(fitted_docs, labels):
    """
    Calculate for each feature the OCFS Score, returned in a vector
    :param fitted_docs: Matrix as returned by fit_transform() in sklearn
    :param labels: Matrix as returned by extract_range() when used to extract the labels
    :return: a vector of OCFS scores
    """
    def calculateMean(label):
        """
        Crude implementation of mean, but faster for sparse
        matrix compared to pandas.DataFrame.mean()
        :param label:
        :return:
        """
        df = docs.loc[docs["Label"] == label]
        df = df.drop(['Label'], axis=1)
        dfLen = len(df)
        mat = df.to_coo().tocsr()
        mean = mat.sum(axis=0, dtype=float)/dfLen
        mean = pd.Series(np.ravel(mean))
        return mean
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
    docs["Label"] = labels
    totalSamples = len(docs)
    # Don't know why mean with dataframe directly yields inaccurate results
    allMean = fitted_docs.mean(axis=0)
    allMean = pd.Series(np.ravel(allMean))
    classgroup = docs.groupby("Label")
    classSamples = docs.groupby("Label").size()
    classMean = {}

    for label in classSamples.index:
        classMean[label] = calculateMean(label)
    ocfs = np.sum(
        [(classSamples[label] / totalSamples) * np.square(classMean[label] - allMean) for label in classSamples.index],
        axis=0)
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

#  DEPRECATED, USE posVectorizer METHOD INSTEAD
# def gen_pos_features(docs, tags, weight): 
#     """
#     Generate POS features for given docs and tags based on specific weighting scheme.
#     :param docs:
#     :param tags:
#     :param weight:
#     :return:
#     """
#     indptr = [0]
#     indices = []
#     data = []
#     vocabulary = {}
#     for d, e in zip(docs, tags):
#         #print(d, len(d), len(e))
#         temp_index = defaultdict(int)
#         for term, pos in zip(d, e):
#             index = vocabulary.setdefault(term, len(vocabulary))
#             temp_index[index] += 1
#             val = weight.setdefault(pos, 0)
#             temp_index[index] = temp_index[index] * val

#         # avoid to create 2 times the same indices within a same document, indices need to be sorted as well
#         for key in sorted(temp_index.keys()):
#             indices.append(key)
#             data.append(temp_index[key])
#         # indptr.append(indptr[-1] + len(e))
#         indptr.append(len(indices))
#     pos_train = csr_matrix((data, indices, indptr), dtype=float)
#     pos_train_normalized = normalize(pos_train, norm='l1', copy=False)
#     # print(temp_index)
#     return pos_train_normalized, vocabulary, pos_train_normalized.shape

#  DEPRECATED, USE posVectorizer METHOD INSTEAD
# def convert(docs, tags, weight, vocabulary, dim):
#     """
#     docstring here
#     :param docs: 
#     :param tags: 
#     :param weight: 
#     :param vocabulary: 
#     :return:
#     """
#     indptr = [0]
#     indices = []
#     data = []
#     column = dim[1]
#     for d, e in zip(docs, tags):
#         #print(d, len(d), len(e))
#         temp_index = defaultdict(int)
#         for term, pos in zip(d, e):
#             if term in vocabulary.keys():
#                 index = vocabulary[term]
#                 temp_index[index] += 1
#                 val = weight.setdefault(pos, 0)
#                 temp_index[index] = temp_index[index] * val

#         # avoid to create 2 times the same indices within a same document, indices need to be sorted as well
#         for key in sorted(temp_index.keys()):
#             indices.append(key)
#             data.append(temp_index[key])
#         # indptr.append(indptr[-1] + len(e))
#         indptr.append(len(indices))
#     pos_train = csr_matrix((data, indices, indptr), shape=(len(docs), column), dtype=float)
#     pos_train_normalized = normalize(pos_train, norm='l1', copy=False)
#     return pos_train_normalized


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
    tempMat.col -= drop_idx.searchsorted(tempMat.col)  # decrease column indices
    tempMat._shape = (tempMat.shape[0], tempMat.shape[1] - len(drop_idx))
    return tempMat.tocsr()


class posVectorizer:
    def __init__(self, weight):
        self.weight = weight

    def _generate_sparse_data(self, docs, tags, fit_flag=True, idx_vocabulary={}):
        indptr = [0]
        indices = []
        data = []
        vocabulary = idx_vocabulary
        for d, e in zip(docs, tags):
            temp_index = defaultdict(int)
            for term, pos in zip(d, e):
                if fit_flag:  # for fitting
                    index = vocabulary.setdefault(term, len(vocabulary))
                    temp_index[index] += 1
                    val = self.weight.setdefault(pos, 0)
                    temp_index[index] = temp_index[index] * val
                else:  # for transform
                    if term in vocabulary.keys():
                        index = vocabulary[term]
                        temp_index[index] += 1
                        val = self.weight.setdefault(pos, 0)
                        temp_index[index] = temp_index[index] * val

            # avoid to create 2 times the same indices within a same document, indices need to be sorted as well
            for key in sorted(temp_index.keys()):
                indices.append(key)
                data.append(temp_index[key])
            # indptr.append(indptr[-1] + len(e))
            indptr.append(len(indices))
        return data, indices, indptr, vocabulary

    def fit(self, docs, tags):
        """
        Generate POS features for given docs and tags based on specific weighting scheme.
        :param docs:
        :param tags:
        :param weight:
        :return:
        """
        data, indices, indptr, self.vocabulary = self._generate_sparse_data(docs, tags)
        posMat = csr_matrix((data, indices, indptr), dtype=float)
        posMat_normalized = normalize(posMat, norm='l1', copy=False)
        self.dim = posMat_normalized.shape
        return posMat_normalized

    def transform(self, docs, tags):
        """
        docstring here
        :param docs:
        :param tags:
        :return:
        """
        column = self.dim[1]
        data, indices, indptr, self.vocabulary = self._generate_sparse_data(docs, tags, fit_flag=False, idx_vocabulary=self.vocabulary)
        posMat = csr_matrix((data, indices, indptr), shape=(len(docs), column), dtype=float)
        posMat_normalized = normalize(posMat, norm='l1', copy=False)
        return posMat_normalized


def main():
    #  parent_dir = Path(__file__).parents[1]
    parent_dir = os.getcwd() # my sys.path is different from PyCharm
    DATA_SET_PATH = os.path.join(parent_dir, "dataset")
    TAGGED_SENTENCES = os.path.join(DATA_SET_PATH, 'text_cleaned_pos.csv')
    LABELS = os.path.join(DATA_SET_PATH, 'shuffled.csv')

    START_RANGE = 0
    END_RANGE = None  # Set to None to get the whole set...

    tagged_sentences = get_tagged_sentences(DATA_SET_PATH, TAGGED_SENTENCES, start_range=START_RANGE,
                                            end_range=END_RANGE, split_pos=False)

    pos_groups = {"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}
    # # tagged_sentences = pre_processing(tagged_sentences, pos_grouping=pos_groups)

    all_docs = []
    all_tags = []

    for tagged_sentence in tagged_sentences:
        words, tags = zip(*tagged_sentence)
        all_docs.append(list(words))
        all_tags.append(list(tags))

    all_labels = get_labels(LABELS, start_range=START_RANGE, end_range=END_RANGE)
    # samplesLen = len(all_labels)
    # testEnd = math.floor(0.3 * samplesLen)  # 20% for test
    # devEnd = math.floor(0.1 * samplesLen)  # 10% for dev

    # DEV_RANGE = (0, devEnd)
    # TEST_RANGE = (devEnd, testEnd)
    # TRAINING_RANGE = (testEnd, samplesLen)

    # dev_docs = extract_range(all_docs, DEV_RANGE[0], DEV_RANGE[1])
    # dev_labels = extract_range(all_labels, DEV_RANGE[0], DEV_RANGE[1])
    # dev_tags = extract_range(all_tags, DEV_RANGE[0], DEV_RANGE[1])
    # docs, tags = get_tagged_sentences(path, TAGGED_SENTENCES,)
    # labels = get_labels(LABELS)
    dataLen = len(all_labels)
    trainEnd = math.floor(0.7 * dataLen)  # 70% for train
    testStart = math.floor(0.8 * dataLen)  # 20% for test
    train_docs, test_docs = all_docs[:trainEnd], all_docs[testStart:]
    train_labels, test_labels = all_labels[:trainEnd], all_labels[testStart:]
    train_tags, test_tags = all_tags[:trainEnd], all_tags[testStart:]

    # work around to prevent scikit performing tokenizing on already tokenized documents...
    bag_of_words = CountVectorizer(
        analyzer='word',
        tokenizer=do_not_tokenize,
        preprocessor=do_not_tokenize,
        token_pattern=None,
        # stop_words="english"
        binary=True  # replaces bow_train = (bow_train >= 1).astype(int)
    )
    bow_train = bag_of_words.fit_transform(train_docs)
    # dev_labels = np.ravel(dev_labels)
    all_labels = np.ravel(all_labels)
    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)
    #
    # print("BOW" , bow_train.shape)
    # print("BOW", bow_train)
    # ocfs = calculate_ocfs_score(bow_train, train_labels)
    # feature_idx = retrieve_features_to_remove(ocfs, 10 ** -7, 10 ** -2)
    # print(bow_train.shape)
    # print(feature_idx)
    # print(len(feature_idx))
    # vocabulary = dict(map(reversed, bag_of_words.vocabulary_.items()))
    # words_to_ignore = [vocabulary[idx] for idx in feature_idx]

    # Removing the features by setting them to zero...
    # pd_bow_train = pd.DataFrame(bow_train.toarray())
    # for idx in feature_idx:
    #     pd_bow_train.iloc[:, idx] = 0
    # pd_bow_train = drop_cols(bow_train, feature_idx)
    # print(pd_bow_train.shape)
    #
    # Train classifier on Bag of Words (Term Presence) and TF-IDF
    # bow_classifier = LogisticRegression(random_state=0, solver='lbfgs',
    #                                     multi_class='multinomial',
    #                                     max_iter=5000
    #                                     ).fit(pd_bow_train, train_labels)
    # bow_train_acc = bow_classifier.score(pd_bow_train, train_labels)
    # pd_bow_test = bag_of_words.transform(test_docs)
    # pd_bow_test = drop_cols(pd_bow_test, feature_idx)
    # bow_test_acc = bow_classifier.score(pd_bow_test, test_labels)

    # print(bow_train_acc)
    # print(bow_test_acc)

    pos_vocab = {'N': 2, 'V': 3, 'A': 4, 'R': 5}  # 5 for N, 3 for V, 2 for A, 1 for R
    # pos_train, word_idx, dim = gen_pos_features(train_docs, train_tags, pos_vocab)
    posFeatures = posVectorizer(pos_vocab)
    pos_train = posFeatures.fit(train_docs, train_tags)
    ocfs_pos = calculate_ocfs_score(pos_train, train_labels)
    pos_feature_idx = retrieve_features_to_remove(ocfs_pos, 10 ** -7, 10 ** -2)
    pd_pos_train = drop_cols(pos_train, pos_feature_idx)
    pos_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                        multi_class='multinomial',
                                        max_iter=5000
                                        ).fit(pd_pos_train, train_labels)
    pos_train_acc = pos_classifier.score(pd_pos_train, train_labels)

    # pos_test = gen_pos_features(test_docs, test_tags, pos_vocab)
    # pos_test = convert(test_docs, test_tags, pos_vocab, word_idx, dim)
    pos_test = posFeatures.transform(test_docs, test_tags)
    pd_pos_test = drop_cols(pos_test, pos_feature_idx)
    pos_test_acc = pos_classifier.score(pd_pos_test, test_labels)
    print(pos_train_acc)
    print(pos_test_acc)


if __name__ == "__main__":
    main()
