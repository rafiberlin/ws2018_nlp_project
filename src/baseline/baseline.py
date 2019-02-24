from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from src.data.helper import get_labels, get_tagged_sentences
import numpy as np
import os
import math
from pathlib import Path

def do_not_tokenize(doc):
    """
    Dummy function to trick scikit vectorizer => avoid tokenizing, processing
    :param doc:
    :return:
    """

    return doc


def main():
    """
    Run the baseline and output results in the console (Accuracy + Macro F1 score for BOW and TFIDF)
    Used to show baseline numbers for the presentation.
    Because we corrected encoding problem on some files afterwards, the numbers may now differ
    :return:
    """

    # Do we still need punctuation removal? at least it can reduce feature space seeing that although
    # the tokenization is good there is still to many useless punctuation.

    parent_dir = Path(__file__).parents[2]
    path = os.path.join(parent_dir, 'dataset', 'processed')
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

    # Test on itself for BoW and TF-IDF
    bow_train_acc = bow_classifier.score(bow_train, train_labels)
    tfidf_train_acc = tf_idf_classifier.score(tfidf_train, train_labels)
    print("Training score BOW", bow_train_acc)
    print("Training score TFIDF", tfidf_train_acc)

    # Convert data into features for testing
    bow_test = bag_of_words.transform(test_docs)
    tfidf_test = tfidf.transform(test_docs)
    test_labels = np.ravel(test_labels)

    # Test on test data
    bow_test_acc = bow_classifier.score(bow_test, test_labels)
    tfidf_test_acc = tf_idf_classifier.score(tfidf_test, test_labels)
    print("Testing score BOW", bow_test_acc)
    print("Testing score TFIDF", tfidf_test_acc)

    # F1 Score for BoW and TF-IDF
    bow_predicted = bow_classifier.predict(bow_test)
    tfidf_predicted = tf_idf_classifier.predict(tfidf_test)
    bow_f1 = f1_score(test_labels, bow_predicted, average=None, labels=['neutral', 'positive', 'negative'])
    tfidf_f1 = f1_score(test_labels, tfidf_predicted, average=None, labels=['neutral', 'positive', 'negative'])
    bow_macro = f1_score(test_labels, bow_predicted, average='macro', labels=['neutral', 'positive', 'negative'])
    tfidf_macro = f1_score(test_labels, tfidf_predicted, average='macro', labels=['neutral', 'positive', 'negative'])
    print("F1 score BOW for neutral, positive, negative", bow_f1)
    print("F1 score TFIDF for neutral, positive, negative", tfidf_f1)
    print("F1 score BOW for macro-average", bow_macro)
    print("F1 score TFIDF for macro-average", tfidf_macro)


if __name__ == "__main__":
    main()
