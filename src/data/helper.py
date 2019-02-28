import pandas as pd
import csv
from nltk.corpus import stopwords as nltk_stopwords
import nltk.corpus.reader.conll as conll
import math
import numpy as np


def extract_range(iterable, start_range=None, end_range=None):
    """
    return a copy of the rows given the start and end range
    :param iterable: an iterable to extract range from
    :param start_range: start extraction at this row
    :param end_range: end extraction at this row
    :return: rows from iterable within the given range
    """
    num_rows = len(iterable)
    if not start_range:
        start_range = 0
    if not end_range:
        end_range = num_rows
    if start_range > end_range:
        start_range = end_range

    return iterable[start_range:end_range]


def get_tagged_sentences(folder, filename, file_extension=".csv", start_range=None, end_range=None, split_pos=True):
    """
    From a csv file create:
    option 1) for each sentence create one list with tuples of (word, pos tag). Return a list of such lists
    option 2) Create two lists. First list: for each sentence create a list of words. Return a list of such lists
                                Second list: for each sentence create a list of pos tags for each word.
                                Return a list of such lists

    :param folder:     Folder to the tagged sentences
    :param filename: the file to parse
    :param file_extension: ending of the file to be parsed
    :param start_range: optional, get sentences from a given index
    :param end_range: optional, get sentences until a given index
    :param split_pos: if false, returns a list of documents, where each of the documents contains a tuple (word,pos),
                      if true 2 separated lists (one list of words, one list of corresponding pos)
    :return: one or 2 lists, see param split_pos
    """
    corpus = conll.ConllCorpusReader(folder, file_extension, ('words', 'pos'))
    tagged_sentences = extract_range(corpus.tagged_sents(filename), start_range, end_range)
    if not split_pos:
        return tagged_sentences

    sentences_only = []
    tags_only = []

    for tagged_sentence in tagged_sentences:
        words, tags = zip(*tagged_sentence)
        # undo tokenize done by ark tagger adding white space, if needed by scikit
        # sentences_only.append(" ".join(list(words)))
        sentences_only.append(list(words))
        tags_only.append(list(tags))
    return sentences_only, tags_only


def get_labels(shuffled_file, start_range=None, end_range=None):
    """
    From a csv file with tweets and their sentiment labels (positive, neutral or negative)
    creates a pandas.dataframe object with labels

    Sentiment labels are encoded as negative =0, positive=1, neutral=2)
    :param shuffled_file: a csv file with 3 columns: 1)index of the tweet starting zero, 2) sentiment label, 3) tweet
    :param start_range: if True: returns labels for tweets starting at this index in the input doc
    :param end_range: if True: returns labels for tweets up to this index in the input doc

    :return: labels as pandas.dataframe objects
    """

    df = pd.read_csv(shuffled_file, sep=',', header=None, names=['ID', 'Label', 'Orig'], quoting=csv.QUOTE_ALL,
                     encoding='utf8')
    df = df.drop(['ID', 'Orig'], axis=1)
    df.replace({'Label': {'negative': 0, 'positive': 1, 'neutral': 2}})

    return extract_range(df, start_range, end_range)


def pre_processing(tagged_sentence, pos_grouping=None,
                   default_pos="DEFAULT",
                   stopwords=None, to_lower=True):
    """
    On pre-tagged sentences apply the following pre-processing steps: 1) Remove stopwords,
                    2) if True: group pos categories in a list to be assigned the same weight during training,
                    3) change sentences to lower case
    :param tagged_sentence: list of lists of tuples (word,pos tag) as returned by get_tagged_sentences
    :param pos_grouping: if True: a dictionary with keys = feature names ("N", "N+R"), values = a list of pos tags
                            to be assigned the same weight during training
    :param default_pos: default pos features
    :param stopwords: set of stopwords to remove
    :param to_lower: boolean value to indicate whether to lower the case of all sentences or not
    :return: list of tagged sentences with removed stopped words, different pos grouping and lower-case sentences
    """
    if pos_grouping is None:
        pos_grouping = {"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}

    if stopwords is None:
        stopwords = set(nltk_stopwords.words('english'))

    processed_sentences = []
    for sentence in tagged_sentence:
        new_sentence = []
        for word_pos in sentence:
            word = word_pos[0]
            pos = word_pos[1]
            group_found = False
            if to_lower:
                word = word.lower()
            if word in stopwords:
                continue
            for pos_group_key, pos_group_values in pos_grouping.items():
                if pos in pos_group_values:
                    new_sentence.append((word, pos_group_key,))
                    group_found = True
                    break
            # Fallback to avoid empty documents.
            # Example tweet that does not contain any of our groups => just emoji + hashtag
            if not group_found:
                new_sentence.append((word, default_pos,))
        processed_sentences.append(new_sentence)
    return processed_sentences


def get_pos_datasets(tagged_sentences, all_labels, pos_groups, percentage_train_data=0.7, percentage_test_data=0.2):
    """
    Applies pre-processing as defined in pre-processing function to the list of tagged sentences,
    splits the dataset into separate dev, training and testing lists,
    splits the labels into dev, training and testing numpy arrays
    The percentage train data starts from the beginning of the list, the percentage test data from the end, such as the
    percentage left in between is reserved for dev purposes.
    Returns all data (docs, labels) as needed by scikit classifiers

    :param tagged_sentences: a list of sentences as lists of tuples (word, pos tag), as created by get_tagged_sentences
    :param all_labels: a pandas data frame object with sentiment labels as created by get_lables
    :param pos_groups: a dictionary with key=name of pos feature,
            value=list of pos tags that will get the same feature weight. E.g. {"A+R": ["A", R"], "N": ["N"]}
    :param percentage_train_data: a floating point number between 0 and 1, percent of data for training
    :param percentage_test_data: a floating point number between 0 and 1, percent of data for testing
    :return: sentences for dev, sentences for training, sentences for testing, labels for dev, labels for training, labels for testing
    """

    processed_tagged_sentences = pre_processing(tagged_sentences, pos_grouping=pos_groups)
    data_len = len(all_labels)
    train_end = math.floor(percentage_train_data * data_len)  # 70% for train
    test_start = math.floor((1.0 - percentage_test_data) * data_len)  # 20% for testing
    dev_docs, train_docs, test_docs = processed_tagged_sentences[train_end:test_start] \
        , processed_tagged_sentences[:train_end] \
        , processed_tagged_sentences[test_start:]
    dev_labels, train_labels, test_labels = all_labels[train_end:test_start] \
        , all_labels[:train_end] \
        , all_labels[test_start:]
    dev_labels = np.ravel(dev_labels)
    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)

    return dev_docs, train_docs, test_docs, dev_labels, train_labels, test_labels


if __name__ == '__main__':
    pass
