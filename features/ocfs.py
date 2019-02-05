import os
from process_data.helper import get_tagged_sentences, get_labels, extract_range
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from baseline.baseline import do_not_tokenize
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import math


def calculate_ocfs_score(fitted_docs, labels):
    """
    Calculate for each feature the OCFS Score, returned in a vector
    :param fitted_docs: Matrix as returned by fit_transform() in sklearn
    :param labels: Matrix as returned by extract_range() when used to extract the labels
    :return: a vector of OCFS scores
    """
    fitted_docs_pd_frame = pd.DataFrame(fitted_docs.toarray())
    fitted_docs_pd_frame["Label"] = labels["Label"]
    all_mean = fitted_docs_pd_frame.mean(axis=0)
    class_mean = fitted_docs_pd_frame.groupby("Label").mean()

    # Sorry for that one, I tried to keep it compact due to memory issues
    # The argument of np.sum is a list of the mean of features for each class
    # when the mean of a featur is calculated for a class, it gets added along the rows (axis=0)
    # to get a feature score for each feature...
    ocfs = np.sum([np.square(class_mean.loc[label,] - all_mean) for label in class_mean.index], axis=0)
    # print(np.square(class_mean.iloc[idx,] - all_mean))
    # print(class_mean.loc[class_mean['Label'] == "positive"])

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

def main():
    parent_dir = Path(__file__).parents[1]
    DATA_SET_PATH = os.path.join(parent_dir, "dataset/raw_data_by_year/train/")
    TAGGED_SENTENCES = os.path.join(DATA_SET_PATH, 'text_cleaned_pos.csv')
    LABELS = os.path.join(DATA_SET_PATH, 'shuffled.csv')

    DEV_RANGE = (0, 6000)
    TEST_RANGE = (DEV_RANGE[1], 18000)
    TRAINING_RANGE = (TEST_RANGE[1], 61212)
    #TRAINING_RANGE = (0, math.floor(0.7 * 61212))
    #DEV_RANGE = (TRAINING_RANGE[1], math.floor(0.8 * 61212))
    #TEST_RANGE = (DEV_RANGE[1], 61212)

    all_docs, all_tags = get_tagged_sentences(DATA_SET_PATH, TAGGED_SENTENCES)
    all_labels = get_labels(shuffled_file=LABELS)

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
    bow_train = bag_of_words.fit_transform(dev_docs)

    ocfs = calculate_ocfs_score(bow_train, dev_labels)
    feature_idx = retrieve_features_to_remove(ocfs, 10 ** -7, 10 ** -2)
    vocabulary = dict(map(reversed, bag_of_words.vocabulary_.items()))
    words_to_ignore = [vocabulary[idx] for idx in feature_idx]

    dev_labels = np.ravel(dev_labels)

    # Removing the features by setting them to zero...
    pd_bow_train = pd.DataFrame(bow_train.toarray())
    for idx in feature_idx:
        pd_bow_train.iloc[:, idx] = 0

    # Train classifier on Bag of Words (Term Presence) and TF-IDF
    bow_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                        multi_class='multinomial',
                                        max_iter=5000
                                        ).fit(bow_train, dev_labels)
    bow_test_acc = bow_classifier.score(pd_bow_train, dev_labels)

    print(bow_test_acc)


if __name__ == "__main__":
    main()