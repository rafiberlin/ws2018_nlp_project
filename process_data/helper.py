import pandas as pd
import csv
from nltk.corpus import stopwords as nltk_stopwords
import nltk.corpus.reader.conll as conll


def extract_range(iterable, start_range=None, end_range=None):
    """
    return a copy of the rows given the start and end range
    :param iterable:
    :param start_range:
    :param end_range:
    :return:
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
    :param folder:     Folder to the tagged sentences
    :param filename: the file to parse
    :param file_extension: ending of the file toi be parsed
    :param start_range: optional, get sentences from a given index
    :param end_range: optional, get sentences until a given index
    :param split_pos: if false, returns a list of documents, where each documents contains a tuple (word,pos),
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
    used to get encoded labels (negative =0, positive 1, neutral 2) from the /dataset/processed/shuffled.csv file
    :param shuffled_file:
    :param start_range:
    :param end_range:
    :return: labels and labels for testing data as pandas.dataframe objects
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
    Apply some pre-processing on pre tagged sentences. Stopwords, POS Grouping, lowering case
    :param tagged_sentence:
    :param pos_grouping:
    :param default_pos:
    :param stopwords:
    :param to_lower:
    :return: tagged sentences with removed stopped words and different pos grouping
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
