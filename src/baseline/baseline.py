from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from data.helper import get_labels, get_tagged_sentences
import numpy as np
import os
import math
from pathlib import Path


def do_not_tokenize(doc):
    """
    Dummy function to trick scikit vectorizer => avoid tokenizing, processing
    :param doc: callable
    :return: callable
    """

    return doc


def print_report(model, test_docs, test_labels, test_name, report_precision):
    """
    Output a well formed report in the console.
    :param model:
    :param test_docs:
    :param test_labels:
    :param test_name:
    :param report_precision:
    :return:
    """
    predicted = model.predict(test_docs)
    testing_accuracy = model.score(test_docs, test_labels)
    print(
        '================================\n\nClassification Report for '
        + test_name
        + ' (Test Data)\n')
    print("\tTesting Accuracy: ", testing_accuracy, "\n")
    print(classification_report(test_labels, predicted, digits=report_precision))


def main(dataset_folder_suffix=None):
    """
    Run the baseline and output results in the console (Accuracy + Macro F1 score for BOW and TFIDF)
    Used to show baseline numbers for the presentation.
    :param dataset_folder_suffix: optional, the suffix for the processed folder. Possible values "no_class_skew", "reshuffled"
    :return: nothing, only print statements
    """

    processed_folder = 'processed'
    if dataset_folder_suffix:
        processed_folder += "_" + dataset_folder_suffix

    parent_dir = Path(__file__).parents[2]
    path = os.path.join(parent_dir, 'dataset', processed_folder)
    tagged_sentences = os.path.join(path, 'text_cleaned_pos.csv')
    label_file = os.path.join(path, 'shuffled.csv')
    docs, tags = get_tagged_sentences(path, tagged_sentences)
    labels = get_labels(label_file)
    data_len = len(labels)
    train_end = math.floor(0.7 * data_len)  # 70% for train
    test_start = math.floor(0.8 * data_len)  # 20% for test
    train_docs, test_docs = docs[:train_end], docs[test_start:]
    train_labels, test_labels = labels[:train_end], labels[test_start:]

    # work around to prevent scikit performing tokenizing on already tokenized documents...
    bag_of_words = CountVectorizer(
        analyzer='word',
        tokenizer=do_not_tokenize,
        preprocessor=do_not_tokenize,
        token_pattern=None,
        # stop_words="english"
        binary=True  # replaces bow_train = (bow_train >= 1).astype(int)
    )

    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=do_not_tokenize,
        preprocessor=do_not_tokenize,
        token_pattern=None,
        # stop_words="english"
    )

    # Convert data into features for training
    bow_train = bag_of_words.fit_transform(train_docs)  # matrix is of type CSR
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

    # Convert data into features for testing
    bow_test = bag_of_words.transform(test_docs)
    tfidf_test = tfidf.transform(test_docs)
    test_labels = np.ravel(test_labels)

    # Test on itself for BoW and TF-IDF
    bow_train_acc = bow_classifier.score(bow_train, train_labels)
    tfidf_train_acc = tf_idf_classifier.score(tfidf_train, train_labels)

    report_precision = 8

    print("\nTraining score BOW", bow_train_acc)
    print_report(bow_classifier, bow_test, test_labels, "BOW", report_precision)
    print("\nTraining score TFIDF", tfidf_train_acc)
    print_report(tf_idf_classifier, tfidf_test, test_labels, "TFIDF", report_precision)


if __name__ == "__main__":
    main()
