import nltk.corpus.reader.conll as conll
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import numpy as np
import os


def get_tagged_sentences(folder, filename, file_extension=".csv", max_rows=40000):
    """

    :param folder:     Folder to the tagged sentences
    :param filename: the file to parse
    :param file_extension: ending of the file toi be parsed
    :return: three lists, one with the tokenized sentences, one with the tags,
            one with tokenized sentences for testing
    """
    corpus = conll.ConllCorpusReader(folder, file_extension, ('words', 'pos'))
    tagged_sentences = corpus.tagged_sents(filename)
    num_rows = 30000

    sentences_only = []
    tags_only = []
    test_sentences = []

    num_sentences = len(tagged_sentences)
    if max_rows > num_sentences or not max_rows:
        max_rows = num_sentences

    for tagged_sentence in tagged_sentences[:num_rows]:
        words, tags = zip(*tagged_sentence)
        # undo tokenize done by ark tagger adding white space, if needed by scikit
        # sentences_only.append(" ".join(list(words)))
        sentences_only.append(list(words))
        tags_only.append(list(tags))

    for tagged_s in tagged_sentences[num_rows:max_rows]:
        words, tags = zip(*tagged_s)
        test_sentences.append(list(words))

    return sentences_only, tags_only, test_sentences


def get_labels(shuffled_file, max_rows=40000):
    """
    used to get encoded labels (negative =0, positive 1, neutral 2) from the /dataset/shuffled.csv file
    :param shuffled_file:
    :param max_rows:
    :return: labels and labels for testing data as pandas.dataframe objects
    """

    df = pd.read_csv(shuffled_file, sep=',', header=None, names=['ID', 'Label', 'Orig'], quoting=csv.QUOTE_ALL,
                     encoding='utf8', nrows=max_rows)
    df = df.drop(['ID', 'Orig'], axis=1)
    labels = df[:30000].replace({'Label': {'negative': 0, 'positive': 1, 'neutral': 2}})
    test_labels = df[30000:40000].replace({'Label': {'negative': 0, 'positive': 1, 'neutral': 2}})
    return labels, test_labels


def do_not_tokenize(doc):
    return doc


if __name__ == "__main__":
    MAIN_FOLDER = "../dataset/"
    path = os.path.join(os.getcwd(), MAIN_FOLDER)
    TAGGED_SENTENCES = os.path.join(path, 'text_cleaned_pos.csv')
    LABELS = os.path.join(path, 'shuffled.csv')
    docs, tags, test_docs = get_tagged_sentences(MAIN_FOLDER, TAGGED_SENTENCES)
    labels, test_labels = get_labels(LABELS)

    # common_stop_words = set(stopwords.words('english'))

    # work around to prevent scikit performing tokenizing on already tokenized documents...
    bag_of_words = CountVectorizer(
        analyzer='word',
        tokenizer=do_not_tokenize,
        preprocessor=do_not_tokenize,
        token_pattern=None,
        # stop_words="english"
    )

    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=do_not_tokenize,
        preprocessor=do_not_tokenize,
        token_pattern=None,
        # stop_words="english"
    )

    bow_classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=700)
    tf_idf_classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=700)
    bow_features = bag_of_words.fit_transform(docs)
    tfidf_features = tfidf.fit_transform(docs)
    class_labels = np.ravel(labels)  # return flat array of labels
    test_class_labels = np.ravel(test_labels)

    #Score on itself: bow
    bow_classifier.fit(bow_features, class_labels)
    acc_train = bow_classifier.score(bow_features, class_labels)
    print("Training score BOW", acc_train)

    # Testing: BoW
    bagOfWords_test = bag_of_words.transform(test_docs)
    X = bagOfWords_test.toarray()
    X = (X > 1).astype(int)
    predictions = bow_classifier.predict(X)
    acc_test = bow_classifier.score(X, test_class_labels)
    #acc_test = bow_classifier.score(predictions, test_class_labels)
    print("Testing score BOW", acc_test)

    #Score on itself: TFIDF
    tf_idf_classifier.fit(tfidf_features, class_labels)
    acc_train = tf_idf_classifier.score(tfidf_features, class_labels)
    print("Training score TFIDF", acc_train)

    # Testing: TF-IDF
    tfidf_test = tfidf.transform(test_docs)
    X = tfidf_test.toarray()
    X = (X > 1).astype(int)
    predictions = tf_idf_classifier.predict(X)
    acc_test = tf_idf_classifier.score(X, test_class_labels)
    #acc_test = bow_classifier.score(predictions, test_class_labels)
    print("Testing score TFIDF", acc_test)
