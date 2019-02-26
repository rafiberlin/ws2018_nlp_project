import os
import math
from data.helper import get_tagged_sentences, get_labels
from data.helper import pre_processing
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from baseline.baseline import do_not_tokenize
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, classification_report
from itertools import product


def drop_cols(matrix, drop_idx):
    """
    Drop column given index to be dropped and a matrix
    :param matrix: input matrix, numpy array
    :param drop_idx: index of a column to be dropped
    :return: a matrix in Compressed Sparse Row format with dropped columns
    """
    drop_idx = np.unique(drop_idx)
    temp_mat = matrix.tocoo()
    keep = ~np.in1d(temp_mat.col, drop_idx)
    temp_mat.data, temp_mat.row, temp_mat.col = temp_mat.data[keep], temp_mat.row[keep], temp_mat.col[keep]
    temp_mat.col -= drop_idx.searchsorted(temp_mat.col)  # decrease column indices
    temp_mat._shape = (temp_mat.shape[0], temp_mat.shape[1] - len(drop_idx))
    return temp_mat.tocsr()


class PosVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, weight):
        """
        Initiate a class object with given weights
        :param weight: a dictionary with keys=pos categories, values=weights
        e.g. {'N': 2, 'V': 3, 'A': 5, "DEFAULT": 1, "E": 4}
        """
        self.weight = weight
        self.vocabulary = defaultdict(int)
        self.dim = None

    def _generate_sparse_data(self, docs, fit_flag=True):
        """
        Generates sparse data matrix, where values of each (term,pos) in each sentence is set according
                            to the weighting scheme, with which PosVectorizer was initialized
        :param docs: list of sentences, which are lists of (word,pos) used for training
        :param fit_flag: if True, fit, if False transform
        :return: sparse data matrix, indices, index pointers
        """
        indptr = [0]
        indices = []
        data = []
        for doc in docs:
            temp_index = defaultdict(int)
            for term, pos in doc:
                word_key = (term, pos)
                # Either fitting or Transforming
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

    def fit(self, docs, train_labels=None):
        """
        Generate POS features for given docs and tags based on specific weighting scheme.
        :param docs: list of sentences, which are lists of (word,pos) used for training
        :param train_labels: parameter required by scikit interface
        :return: model parameters according to the given training data
        """

        self.vocabulary.clear()
        data, indices, indptr = self._generate_sparse_data(docs)
        pos_mat = csr_matrix((data, indices, indptr), dtype=float)
        pos_mat_normalized = normalize(pos_mat, norm='l1', copy=False)
        self.dim = pos_mat_normalized.shape
        return self

    def transform(self, docs):
        """
        Reduces matrix to its most important features.
        :param docs: list of sentences, which are lists of (word,pos) used for training
        :return: normalized matrix in Compressed Sparse Row format
        """
        column = self.dim[1]
        data, indices, indptr = self._generate_sparse_data(docs, fit_flag=False)
        pos_mat = csr_matrix((data, indices, indptr), shape=(len(docs), column), dtype=float)
        pos_mat_normalized = normalize(pos_mat, norm='l1', copy=False)
        return pos_mat_normalized


class OCFS(BaseEstimator, TransformerMixin):
    def __init__(self, number_to_delete=10000):
        """
        Constructor, initialized with the number of features to cut off
        :param number_to_delete: based on the number of features used during fit(),
        converts the number_to_delete to a percentile,
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

        def _calculate_mean(class_label):
            """
            Crude implementation of mean, but faster for sparse
            matrix compared to pandas.DataFrame.mean()
            :param class_label:
            :return: mean value of all the observations in the dataframe (scalar value)
            """
            df = docs.loc[docs["Label"] == class_label]
            df = df.drop(['Label'], axis=1)
            df_len = len(df)
            mat = df.to_coo().tocsr()
            mean = mat.sum(axis=0, dtype=float) / df_len
            mean = pd.Series(np.ravel(mean))
            return mean

        docs = pd.SparseDataFrame(fitted_docs)
        docs = docs.fillna(0)
        docs["Label"] = labels
        total_samples = len(docs)
        # Don't know why mean with dataframe directly yields inaccurate results
        all_mean = fitted_docs.mean(axis=0)
        all_mean = pd.Series(np.ravel(all_mean))
        docs.groupby("Label")
        class_samples = docs.groupby("Label").size()
        class_mean = {}

        for label in class_samples.index:
            class_mean[label] = _calculate_mean(label)
        # Multiply by 1000 to avoid underflow
        ocfs = np.sum(
            [1000 * (class_samples[label] / total_samples) * np.square(class_mean[label] - all_mean) for label in
             class_samples.index],
            axis=0)
        return ocfs

    def fit(self, pos_train, train_labels):
        """
        Creates a list of features to remove
        :param pos_train: normalized matrix in Compressed Sparse Row format, as returned from transform function of
                                                        PosVectorizer class
        :param train_labels: pandas padaframe of corresponding labels
        :return: list of features to delete
        """

        ocfs_pos = OCFS._calculate_ocfs_score(pos_train, train_labels)
        number_of_feature = ocfs_pos.shape[0]

        # non_zero = np.count_nonzero(ocfs_pos)
        # print("Number of zero values after OCFS:" + str(number_of_feature - non_zero))

        if number_of_feature < self.number_to_delete:
            self.number_to_delete = number_of_feature
        percentile = int(round((self.number_to_delete / number_of_feature) * 100))
        lower_bound = np.percentile(ocfs_pos, percentile)
        self.feature_to_delete = [idx for idx in range(number_of_feature) if lower_bound > ocfs_pos[idx]]
        return self

    def transform(self, pos_train):
        """
        Removes features that are under the cutoff mark
        :param pos_train: normalized matrix in Compressed Sparse Row format, as returned from transform function of
                                                        PosVectorizer class
        :return: matrix with removed features
        """
        return drop_cols(pos_train, self.feature_to_delete)


def create_pos_weight_combination(pos_groups, weighing_scale):
    """
    Creates a list of all possible combinations of weights for the given pos groups and a weighting scale
    e.g.: pos groups.keys = [V,A,N,E], weighting scale=4
    [{'V': 1, 'A': 1, 'N': 1, 'E': 1}, {'V': 1, 'A': 1, 'N': 1, 'E': 2}, ....]

    :param pos_groups: dictionary of keys=pos categories, values=their weights
     e.g. {"V": ["V"], "A": ["A", "R"], "N": ["N"], "E": ["E"]}
    :param weighing_scale: integer scale, from 1 to the value in weighing scale
    :return: list of dictionaries, key=pos category, value=its weight
    """
    group_keys = pos_groups.keys()
    weights = list(range(1, weighing_scale + 1))
    return [dict(zip(group_keys, list(combi))) for combi in product(set(weights), repeat=len(group_keys))]


def main():
    """
    Creates two models: BoW and Pos with ocfs
    Creates a pipeline with union of two models.
    Trains and scores the models in the pipeline.
    """
    parent_dir = Path(__file__).parents[1]
    # parent_dir = os.getcwd() # my sys.path is different from PyCharm
    data_set_path = os.path.join(parent_dir, "dataset", "processed")
    tagged_sentences = os.path.join(data_set_path, 'text_cleaned_pos.csv')
    labels = os.path.join(data_set_path, 'shuffled.csv')

    start_range = 0
    end_range = None  # Set to None to get the whole set...
    tagged_sentences = get_tagged_sentences(data_set_path, tagged_sentences, start_range=start_range,
                                            end_range=end_range, split_pos=False)

    # pos_groups = {"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}
    pos_groups = {"V": ["V"], "A": ["A", "R"], "N": ["N"], "E": ["E"]}
    tagged_sentences = pre_processing(tagged_sentences, pos_grouping=pos_groups)

    all_labels = get_labels(labels, start_range=start_range, end_range=end_range)

    data_len = len(all_labels)
    train_end = math.floor(0.7 * data_len)  # 70% for train
    train_start = math.floor(0.8 * data_len)  # 20% for test
    number_of_features_to_delete = 35000

    train_docs, test_docs = tagged_sentences[:train_end], tagged_sentences[train_start:]
    train_labels, test_labels = all_labels[:train_end], all_labels[train_start:]

    # dev_labels = np.ravel(dev_labels)
    # all_labels = np.ravel(all_labels)
    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)

    # work around to prevent scikit performing tokenizing on already tokenized documents...
    bag_of_words = CountVectorizer(
        analyzer='word',
        tokenizer=do_not_tokenize,
        preprocessor=do_not_tokenize,
        binary=True  # replaces bow_train = (bow_train >= 1).astype(int)
    )

    bow_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                        multi_class='multinomial',
                                        max_iter=5000
                                        )

    bow_pipeline = Pipeline([
        ('bowweighing', bag_of_words),
        ('bowclassifier', bow_classifier),
    ])

    # 5 for N, 3 for V, 2 for A, 1 for R
    # pos_vocab = {'N': 2, 'V': 3, 'A': 4, 'R': 5}

    pos_vocab = {'N': 2, 'V': 3, 'A': 5, "DEFAULT": 1, "E": 4}
    maxent_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                           multi_class='multinomial',
                                           max_iter=5000
                                           )

    pos_feature = PosVectorizer(pos_vocab)
    pos_train = pos_feature.fit_transform(train_docs)
    ocfs = OCFS(number_of_features_to_delete)
    ocfs.fit(pos_train, train_labels)
    pd_pos_train = ocfs.transform(pos_train)
    maxent_classifier.fit(pd_pos_train, train_labels)

    # Manual Steps, should deliver the same results as a pipeline
    # pos_test = pos_feature.transform(test_docs)
    # pd_pos_test = ocfs.transform(pos_test)
    # pos_train_acc = maxent_classifier.score(pd_pos_train, train_labels)
    # pos_test_acc = maxent_classifier.score(pd_pos_test, test_labels)
    # print("Manual Steps", pos_train_acc)
    # print("Manual Test", pos_test_acc)

    unified_pipeline = Pipeline([
        # Use FeatureUnion to combine the features from bow and pos
        ('union', FeatureUnion(
            transformer_list=[
                # Pipeline for pulling features from the post's subject line
                ('bow', Pipeline([
                    ('bowweighing', bag_of_words),
                ])),
                # Pipeline for standard bag-of-words model for body
                ('pos', Pipeline([
                    ('posweighing', PosVectorizer(pos_vocab)),
                    ('ocfs', OCFS(number_of_features_to_delete)),
                ])),
            ],
            # weight components in FeatureUnion
            transformer_weights={
                'bow': 0.7,
                'pos': 0.3,
            },
        )),
        # Use a MaxEnt classifier on the combined features
        ('classifier', maxent_classifier),
    ])

    unified_pipeline.fit(train_docs, train_labels)
    pos_train_acc_unified_pipeline = unified_pipeline.score(train_docs, train_labels)
    pos_test_acc_unified_pipeline = unified_pipeline.score(test_docs, test_labels)

    unified_predicted = unified_pipeline.predict(test_docs)
    unified_f1 = f1_score(test_labels, unified_predicted, average="macro", labels=['neutral', 'positive', 'negative'])
    print("Unified Pipeline Train", pos_train_acc_unified_pipeline)
    print("Unified Pipeline Test", pos_test_acc_unified_pipeline)
    print("Unified Pipeline Test F1", unified_f1)
    classification_report(unified_predicted, test_labels)


if __name__ == "__main__":
    main()
