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


class PosVectorizer:
    def __init__(self, weight):
        self.weight = weight
        self.vocabulary = defaultdict(int)
        self.dim = None

    def _generate_sparse_data(self, docs, tags, fit_flag=True):
        indptr = [0]
        indices = []
        data = []
        for d, e in zip(docs, tags):
            temp_index = defaultdict(int)
            for term, pos in zip(d, e):
                word_key = (term, pos)
                ## Either fitting or Transforming
                if fit_flag or (not fit_flag and word_key in self.vocabulary.keys()):
                    index = self.vocabulary.setdefault(word_key, len(self.vocabulary))
                    val = self.weight.setdefault(pos, 0)
                    temp_index[index] += val

            # avoid to create 2 times the same indices within a same document, indices need to be sorted as well
            for key in sorted(temp_index.keys()):
                indices.append(key)
                data.append(temp_index[key])
            indptr.append(len(indices))
        return data, indices, indptr

    def fit(self, docs, tags):
        """
        Generate POS features for given docs and tags based on specific weighting scheme.
        :param docs:
        :param tags:
        :param weight:
        :return:
        """
        self.vocabulary.clear()
        data, indices, indptr = self._generate_sparse_data(docs, tags)
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
        data, indices, indptr = self._generate_sparse_data(docs, tags, fit_flag=False)
        posMat = csr_matrix((data, indices, indptr), shape=(len(docs), column), dtype=float)
        posMat_normalized = normalize(posMat, norm='l1', copy=False)
        return posMat_normalized


class OCFS:
    def __init__(self, number_to_delete=10000):
        """
        Constructor
        :param number_to_delete: based on the number of features used during fit(),
        converts the number_to_delete to  a percentile,
        which is used to define the number of features to be removed
        """
        self.number_to_delete = number_to_delete
        self.feature_to_delete = []

    @classmethod
    def _calculate_ocfs_score(cls, fitted_docs, labels):
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
            mean = mat.sum(axis=0, dtype=float) / dfLen
            mean = pd.Series(np.ravel(mean))
            return mean

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
        # Multiply by 1000 to avoid underflow
        ocfs = np.sum(
            [1000 * (classSamples[label] / totalSamples) * np.square(classMean[label] - allMean) for label in
             classSamples.index],
            axis=0)
        return ocfs

    @classmethod
    def _retrieve_features_to_remove(self, ocfs, lowest_val, highest_val):
        """
        Return the index of the features to discard, based on some boundary values
        :param ocfs:
        :param lowest_val:
        :param highest_val:
        :return:
        """

        return [idx for idx, val in enumerate(ocfs) if val < lowest_val or val > highest_val]

    def fit(self, pos_train, train_labels):
        """

        :param pos_train:
        :param train_labels:
        :return:
        """

        ocfs_pos = OCFS._calculate_ocfs_score(pos_train, train_labels)
        number_of_feature = ocfs_pos.shape[0]
        if number_of_feature < self.number_to_delete:
            number_of_feature = self.number_to_delete
        percentile = int(round((self.number_to_delete / number_of_feature) * 100))
        lower_bound = np.percentile(ocfs_pos, percentile)
        self.feature_to_delete = [idx for idx in range(ocfs_pos.shape[0]) if lower_bound > ocfs_pos[idx]]
        return self

    def transform(self, pos_train):
        return drop_cols(pos_train, self.feature_to_delete)


def main():
    parent_dir = Path(__file__).parents[1]
    # parent_dir = os.getcwd() # my sys.path is different from PyCharm
    DATA_SET_PATH = os.path.join(parent_dir, "dataset")
    TAGGED_SENTENCES = os.path.join(DATA_SET_PATH, 'text_cleaned_pos.csv')
    LABELS = os.path.join(DATA_SET_PATH, 'shuffled.csv')

    START_RANGE = 0
    END_RANGE = 30000  # Set to None to get the whole set...
    NUMBER_OF_FEATURES_TO_DELETE = 30000
    tagged_sentences = get_tagged_sentences(DATA_SET_PATH, TAGGED_SENTENCES, start_range=START_RANGE,
                                            end_range=END_RANGE, split_pos=False)

    pos_groups = {"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}
    tagged_sentences = pre_processing(tagged_sentences, pos_grouping=pos_groups)

    all_docs = []
    all_tags = []

    for tagged_sentence in tagged_sentences:
        words, tags = zip(*tagged_sentence)
        all_docs.append(list(words))
        all_tags.append(list(tags))

    all_labels = get_labels(LABELS, start_range=START_RANGE, end_range=END_RANGE)

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

    # dev_labels = np.ravel(dev_labels)
    all_labels = np.ravel(all_labels)
    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)

    # bow_train = bag_of_words.fit_transform(train_docs)
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
    posFeatures = PosVectorizer(pos_vocab)
    pos_train = posFeatures.fit(train_docs, train_tags)

    ocfs = OCFS(NUMBER_OF_FEATURES_TO_DELETE)
    ocfs.fit(pos_train, train_labels)
    pd_pos_train = ocfs.transform(pos_train)

    pos_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                        multi_class='multinomial',
                                        max_iter=5000
                                        ).fit(pd_pos_train, train_labels)
    pos_train_acc = pos_classifier.score(pd_pos_train, train_labels)

    pos_test = posFeatures.transform(test_docs, test_tags)
    pd_pos_test = ocfs.transform(pos_test)
    pos_test_acc = pos_classifier.score(pd_pos_test, test_labels)
    print(pos_train_acc)
    print(pos_test_acc)
    print(pd_pos_train.shape)
    print(pd_pos_test.shape)


if __name__ == "__main__":
    main()
