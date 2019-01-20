import nltk.corpus.reader.conll as conll
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import numpy as np



def get_tagged_sentences(folder, filename, file_extension=".csv", max_rows=20000):
    """

    :param folder:     Folder to the tagged sentences
    :param filename: the file to parse
    :param file_extension: ending of the file toi be parsed
    :return: two lists, one with the tokenized sentences, one with the tags
    """
    corpus = conll.ConllCorpusReader(folder, file_extension, ('words', 'pos'))
    tagged_sentences = corpus.tagged_sents(filename)

    sentences_only = []
    tags_only = []

    num_sentences = len(tagged_sentences)
    if max_rows > num_sentences or not max_rows:
        max_rows = num_sentences

    for tagged_sentence in tagged_sentences[:max_rows]:
        words, tags = zip(*tagged_sentence)
        #undo tokenize done by ark tagger adding white space, if needed by scikit
        #sentences_only.append(" ".join(list(words)))
        sentences_only.append(list(words))
        tags_only.append(list(tags))
    return sentences_only, tags_only


def get_labels(shuffled_file, max_rows=20000):
    """
    used to get encoded labels (negative =0, positive 1, neutral 2) from the /dataset/shuffled.csv file
    :param shuffled_file:
    :param max_rows:
    :return:
    """

    df = pd.read_csv(shuffled_file, sep=',', header=None, names=['ID', 'Label', 'Orig'], quoting=csv.QUOTE_ALL,
                     encoding='utf8', nrows=max_rows)
    df = df.drop(['ID', 'Orig'], axis=1)
    labels = df.replace({'Label': {'negative': 0, 'positive': 1, 'neutral': 2}})
    return labels


def do_not_tokenize(doc):
    return doc


TAGGED_SENTENCES = "../dataset/text_cleaned_pos.csv"
MAIN_FOLDER = "../dataset/"
LABELS = MAIN_FOLDER + "shuffled.csv"
docs, tags = get_tagged_sentences(MAIN_FOLDER, TAGGED_SENTENCES)
labels = get_labels(LABELS)

common_stop_words = set(stopwords.words('english'))

#work around to prevent scikit performing tokenizing on already tokenized documents...
bag_of_words = CountVectorizer(
    analyzer='word',
    tokenizer=do_not_tokenize,
    preprocessor=do_not_tokenize,
    token_pattern=None,
    #stop_words="english"
)

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=do_not_tokenize,
    preprocessor=do_not_tokenize,
    token_pattern=None,
    #stop_words="english"
)

bow_classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=700)
tf_idf_classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=700)
bow_features = bag_of_words.fit_transform(docs)
tfidf_features = tfidf.fit_transform(docs)
class_labels = np.ravel(labels)

bow_classifier.fit(bow_features, class_labels)
acc_train = bow_classifier.score(bow_features, class_labels)
print("Training score BOW", acc_train)


tf_idf_classifier.fit(tfidf_features, class_labels)
acc_train = tf_idf_classifier.score(tfidf_features, class_labels)
print("Training score TFIDF", acc_train)