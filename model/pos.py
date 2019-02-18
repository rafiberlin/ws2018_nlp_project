import os
import sys
from features.ocfs import *

sys.path.insert(0, os.getcwd())
import math
from process_data.helper import get_tagged_sentences, get_labels
from process_data.spell_correction import pre_processing
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from baseline.baseline import do_not_tokenize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score, classification_report
import random
from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Pool


def return_best_pos_weight(tagged_sentences, all_labels, pos_groups, weighing_scale, features_to_remove,
                           union_transformer_weights=None, percentage_train_data=0.7, percentage_test_data=0.2,
                           use_multi_processing=False):
    """

    :param tagged_sentences:
    :param all_labels:
    :param pos_groups:
    :param weighing_scale:
    :param features_to_remove:
    :param union_transformer_weights:
    :param percentage_train_data:
    :param percentage_test_data:
    :return:
    """

    if union_transformer_weights is None:
        union_transformer_weights = {'bow': 0.7, 'pos': 0.3, }
    weights = union_transformer_weights

    processed_tagged_sentences = pre_processing(tagged_sentences, pos_grouping=pos_groups)

    # debugging multithread
    # all_pos_vocab = create_pos_weight_combination(pos_groups, weighing_scale)[:4]
    all_pos_vocab = create_pos_weight_combination(pos_groups, weighing_scale)
    data_len = len(all_labels)
    train_end = math.floor(percentage_train_data * data_len)  # 70% for train
    train_start = math.floor((1.0 - percentage_test_data) * data_len)  # 20% for testing
    train_docs, test_docs = processed_tagged_sentences[:train_end], processed_tagged_sentences[train_start:]
    train_labels, test_labels = all_labels[:train_end], all_labels[train_start:]
    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)

    # Process the model training with all combination in parallel, letting one core for CPU
    if use_multi_processing:
        cpu_cores = mp.cpu_count()
    else:
        cpu_cores = 1
    original_size = len(all_pos_vocab)
    middle = original_size // cpu_cores
    # fix when only one combination is available
    if middle == 0:
        middle = original_size

    list_of_jobs = split_list(all_pos_vocab, middle)
    num_jobs = len(list_of_jobs)

    args = [[train_docs, test_docs, train_labels, test_labels, features_to_remove, weights, job] for job in
            list_of_jobs]

    with Pool(num_jobs) as p:
        results = p.map(argument_wrapper_for_run_model_for_all_combination,
                        args)
        p.close()
        p.join()

    return results


def split_list(the_list, chunk_size):
    """
    From https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

    :param the_list:
    :param chunk_size:
    :return:
    """
    result_list = []
    while the_list:
        result_list.append(the_list[:chunk_size])
        the_list = the_list[chunk_size:]

    return result_list


def run_model_for_all_combination(train_docs, test_docs, train_labels, test_labels, features_to_remove, weights,
                                  all_pos_vocab):
    best_model_parameters = []
    i = 1
    for pos_vocab in all_pos_vocab:
        print("Round:" + str(i))
        i += 1
        scores = run_pos_model(train_docs, test_docs, train_labels, test_labels, pos_vocab,
                               number_of_features_to_delete=features_to_remove,
                               union_transformer_weights=weights)
        # print("Result for ", pos_vocab, scores)
        if scores is not None:
            best_model_parameters.append((pos_vocab, scores,))

    return best_model_parameters


def run_pos_model(train_docs, test_docs, train_labels, test_labels, pos_vocab, number_of_features_to_delete=30000,
                  union_transformer_weights=None,
                  accuracy_to_beat=0.0, f1_score_to_beat=0.0):
    """

    :param train_docs:
    :param test_docs:
    :param train_labels:
    :param test_labels:
    :param pos_vocab:
    :param number_of_features_to_delete:
    :param union_transformer_weights:
    :param accuracy_to_beat: Reminder for BOW, 0.6252552478967573
    :param f1_score_to_beat: Reminder for BOW, 0.5865028050367952):
    :return:
    """

    if union_transformer_weights is None:
        union_transformer_weights = {'bow': 0.7, 'pos': 0.3, }

    maxent_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                           multi_class='multinomial',
                                           max_iter=5000
                                           )

    bag_of_words = CountVectorizer(
        analyzer='word',
        tokenizer=do_not_tokenize,
        preprocessor=do_not_tokenize,
        binary=True  # replaces bow_train = (bow_train >= 1).astype(int)
    )

    pos_vectorizer = PosVectorizer(pos_vocab)

    pos_bow_pipeline = Pipeline([
        # Use FeatureUnion to combine the features from bow and pos
        ('union', FeatureUnion(
            transformer_list=[
                # Pipeline for pulling features from the post's subject line
                ('bow', Pipeline([
                    ('bowweighing', bag_of_words),
                ])),
                # Pipeline for standard bag-of-words model for body
                ('pos', Pipeline([
                    ('posweighing', pos_vectorizer),
                    ('ocfs', OCFS(number_of_features_to_delete)),
                ])),
            ],
            # weight components in FeatureUnion
            transformer_weights=union_transformer_weights,
        )),
        # Use a MaxEnt classifier on the combined features
        ('classifier', maxent_classifier),
    ])

    pos_bow_pipeline.fit(train_docs, train_labels)
    pos_train_acc_unified_pipeline = pos_bow_pipeline.score(train_docs, train_labels)
    pos_test_acc_unified_pipeline = pos_bow_pipeline.score(test_docs, test_labels)

    unified_predicted = pos_bow_pipeline.predict(test_docs)
    unified_f1 = f1_score(test_labels, unified_predicted, average="macro", labels=['neutral', 'positive', 'negative'])

    if accuracy_to_beat < pos_test_acc_unified_pipeline or f1_score_to_beat < unified_f1:
        return (pos_train_acc_unified_pipeline, pos_test_acc_unified_pipeline, unified_f1,)


def argument_wrapper_for_run_model_for_all_combination(args):
    return run_model_for_all_combination(*args)


if __name__ == "__main__":
    pass
