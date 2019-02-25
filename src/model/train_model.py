import src.features.ocfs as ocfs
from src.data.helper import get_pos_datasets
from sklearn.feature_extraction.text import CountVectorizer
from src.baseline.baseline import do_not_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score
import multiprocessing as mp
from multiprocessing import Pool
from pickle import dump as p_dump
from pickle import load as p_load


def return_best_pos_weight(tagged_sentences, all_labels, pos_groups, weighing_scale, features_to_remove,
                           union_transformer_weights=None, percentage_train_data=0.7, percentage_test_data=0.2,
                           use_multi_processing=False):
    """

    :param tagged_sentences: list of sentences as lists of tuples as created by get_tagged_sentences
    :param all_labels: a pandas data frame object with labels for tagged sentences
    :param pos_groups: dict with keys=feature names, values=list of pos tags that have this feature
    :param weighing_scale: int, from 1 to this number is the scale for assigning weigths to features
    :param features_to_remove: int, number for featrue cutoff with ocfs technique
    :param union_transformer_weights: dict, key=name of model (e.g.pos,bow), value=float between 0 and 1,
                                                    weight of this model in training
    :param percentage_train_data: float between 0 and 1, percentage of training data
    :param percentage_test_data: float between 0 and 1, percentage of test data
    :param use_multi_processing: boolean: True = use multiprocessing for training
    :return: list of best scoring assignments of weights to featrues
                e.g. of one entry in the list ({'V': 2, 'A': 4, 'N+#': 1, 'R': 1, 'E': 4, 'DEFAULT': 0},
                    (0.8574962658700522, 0.627297231070816, 0.5884891927065569))
    """

    if union_transformer_weights is None:
        union_transformer_weights = {'bow': 0.7, 'pos': 0.3, }
    weights = union_transformer_weights

    # debugging multithread
    # all_pos_vocab = ocfs.create_pos_weight_combination(pos_groups, weighing_scale)[:4]
    all_pos_vocab = ocfs.create_pos_weight_combination(pos_groups, weighing_scale)

    train_docs, test_docs, train_labels, test_labels = get_pos_datasets(tagged_sentences, all_labels,
                                                                        pos_groups, percentage_train_data,
                                                                        percentage_test_data)

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
    Splits a list into evenly sized chunks

    :param the_list: list to split
    :param chunk_size: number of elements in one chunk
    :return: a list of lists, which are of the same size
    """
    result_list = []
    while the_list:
        result_list.append(the_list[:chunk_size])
        the_list = the_list[chunk_size:]

    return result_list


def run_model_for_all_combination(train_docs, test_docs, train_labels, test_labels, features_to_remove, weights,
                                  all_pos_vocab):
    """

    :param train_docs: lists of sentences as lists of tuples (word, pos) used for training
    :param test_docs: lists of sentences as lists of tuples (word, pos) used for testing
    :param train_labels: pandas data frame object with sentiment labels for training docs
    :param test_labels: pandas data frame object with sentiment labels for testing docs
    :param features_to_remove: numebr of features to delete with ocfs feature selection
    :param weights: dict, key=name of model (e.g.pos,bow), value=float between 0 and 1,
                                                    weight of this model in training
    :param all_pos_vocab: list of dictionaries. Each dict: with feature names as keys and pos categories that
                                                have this feature as values
    :return: a set of three scores: accuracy for training, for testing , and unified f1 for testing
    """
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

    :param train_docs: lists of sentences as lists of tuples (word, pos) used for training
    :param test_docs: lists of sentences as lists of tuples (word, pos) used for testing
    :param train_labels: pandas data frame object with sentiment labels for training docs
    :param test_labels: pandas data frame object with sentiment labels for testing docs
    :param pos_vocab: dict with feature names as keys and some values. Values will be overwritten with a list created
                                                                        from key
    :param number_of_features_to_delete: numebr of features to delete with ocfs feature selection
    :param union_transformer_weights: dict with keys= model names (bow,pos) and values = floats between 0 and 1,
                                        weights for each model in model union
    :param accuracy_to_beat: Reminder for BOW, 0.6252552478967573
    :param f1_score_to_beat: Reminder for BOW, 0.5865028050367952):
    :return: a set three scores: accuracy for training, for testing , and unified f1 for testing
    """

    pos_bow_pipeline = create_fitted_model(train_docs, train_labels, pos_vocab, number_of_features_to_delete,
                                           union_transformer_weights)

    pos_train_acc_unified_pipeline = pos_bow_pipeline.score(train_docs, train_labels)
    pos_test_acc_unified_pipeline = pos_bow_pipeline.score(test_docs, test_labels)

    unified_predicted = pos_bow_pipeline.predict(test_docs)
    unified_f1 = f1_score(test_labels, unified_predicted, average="macro", labels=['neutral', 'positive', 'negative'])

    if accuracy_to_beat < pos_test_acc_unified_pipeline or f1_score_to_beat < unified_f1:
        return (pos_train_acc_unified_pipeline, pos_test_acc_unified_pipeline, unified_f1,)


def save_model(classifier, file_name):
    """
    Wrapper for pickle.dump (creates the file object needed from the string)
    :param classifier: pos bow pipeline model
    :param file_name: under which to sace the model
    :return: nothing, just saves the object
    """
    with open(file_name, 'wb') as file:
        p_dump(classifier, file)


def load_model(file_name):
    """
    Wrapper for pickle.load. File object was created from the string
    :param file_name: name of file where the model is saved
    :return: the model
    """
    with open(file_name, 'rb') as file:
        classifier = p_load(file)
    return classifier


def create_fitted_model(train_docs, train_labels, pos_vocab, number_of_features_to_delete=30000,
                        union_transformer_weights=None,
                        ):
    """
    Creates model from given parameters

    :param train_docs: list of sentences as lists of tuples (word,pos) used for training
    :param train_labels: pandas dataframe object with labels for train docs
    :param pos_vocab: dict, one weighting combination. Key=pos category, value=weight
           e.g. {'V': 1, 'A': 1, 'N': 1, 'E': 1}
    :param number_of_features_to_delete: int, num of featrues to delete
    :param union_transformer_weights: dict, key=name of model (e.g.pos,bow), value=float between 0 and 1,
                                                    weight of this model in training
    :return: fitted model

    """

    # from pickle import dump, load
    # print("Test")
    # dump(pos_bow_pipeline, open('filename.joblib','wb'))
    # clf2 = load(open('filename.joblib','rb'))
    # res = clf2.predict(test_docs)
    #
    # print(res)
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

    pos_vectorizer = ocfs.PosVectorizer(pos_vocab)

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
                    ('ocfs', ocfs.OCFS(number_of_features_to_delete)),
                ])),
            ],
            # weight components in FeatureUnion
            transformer_weights=union_transformer_weights,
        )),
        # Use a MaxEnt classifier on the combined features
        ('classifier', maxent_classifier),
    ])

    pos_bow_pipeline.fit(train_docs, train_labels)

    return pos_bow_pipeline


def argument_wrapper_for_run_model_for_all_combination(args):
    return run_model_for_all_combination(*args)


if __name__ == "__main__":
    pass
