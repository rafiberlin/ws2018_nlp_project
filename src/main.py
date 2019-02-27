import os
import sys
import time
from model.train_model import return_best_pos_weight, create_fitted_model, save_model, load_model
from data.helper import get_tagged_sentences, get_labels, get_pos_datasets
from pathlib import Path
import ast
from sklearn.metrics import f1_score


def get_pos_groups_from_vocab(pos_vocab):
    """
    Creates dictionary with key= feature name, value=pos tags from feature name split into a list of tags
    Assumption: Key for merged groups A and V is A+V.
    :param pos_vocab: dict with feature names as keys and some values. Values will be overwritten with a list created
                                                                        from key
    :return: dictionary with key=feature name with pos tags separated by +, value=list of pos tags with this feature
    """
    return {key: key.split("+") for key in pos_vocab.keys()}


def save_results(result_path, filename, results):
    """
    Write the results of testing into a .txt file
    :param result_path: path where the file with results will be stores
    :param filename: name of file where to store the results
    :param results: list of results
    :return: nothing, just write into a file during the function execution
    """

    # Save results
    orig_stdout = sys.stdout
    output = os.path.join(result_path, filename)
    with open(output, 'w') as file:
        sys.stdout = file
        for item in results:
            print(item)
    sys.stdout = orig_stdout


def create_prefix_for_model_persistence(p_vocab,
                                        f_to_delete,
                                        u_weights,
                                        train_percent):
    """
    Create a prefix for the default execution of main.py based on the parameters. The prefix is used to name the file.
    :param p_vocab: dict,
    :param f_to_delete: int, feature cutoff number
    :param u_weights: float between 0 and 1 for union of two models: how much weight bow and pos have
    :param train_percent: float between 0 and 1, percentage of dataset used for training
    :return: string, a prefix to be used as a file name
    """
    pref = ""
    for pos in sorted(p_vocab.keys()):
        pref += pos + str(p_vocab[pos]) + "_"
    pref += str(f_to_delete) + "_" + str(u_weights["bow"]) + "_" + str(u_weights["pos"]) + "_" + str(train_percent)
    return pref


def create_prefix(p_groups,
                  w_scale,
                  f_to_delete,
                  u_weights,
                  training_percent,
                  test_percent):
    """
    Creates a prefix based on the user-defined non-default model parameters, for file-naming
    :param p_groups: dict with keys=feature names, values=list of pos tags with that feature weight
    :param w_scale: int, weights scale between 1 and teh given number
    :param f_to_delete: int, number of featrues to delete
    :param u_weights: dict with keys= model names (bow,pos) and values = floats between 0 and 1, weights of the model
    :param training_percent: float between 0 and 1, percent of data used for training
    :param test_percent: float between 0 and 1, percent of data used for testing
    :return: returns a string to be used as a file name, indicating parameters used for a model
    """
    prefix_group = "_".join(["-".join(value) for value in sorted(p_groups.values())])
    union_weight_prefix = str(u_weights["bow"]) + "_" + str(u_weights["pos"])
    training = str(training_percent) + "_" + str(test_percent)
    prefix = prefix_group + "_" + str(
        w_scale) + "_" + str(f_to_delete) + "_" + union_weight_prefix + "_" + training
    return prefix


def run_logic(tagged_sentences, all_labels, pos_groups, weighing_scale, feature_to_delete,
              union_weights, training_percent, test_percent, split_job, result_folder):
    """
    Run the main logic of the project given the user-defined non-default parameters

    :param tagged_sentences: list of sentences as lists of tuples (word, pos) as returned by get_tagged_sentences
    :param all_labels: pandas data frame object with sentiment labels for pos-tagged sentences
    :param pos_groups: dict with keys=names of pos features, values=list of pos categories to have that featrue
    :param weighing_scale: int, from 1 to this number is the weighting scale to assign to features
    :param feature_to_delete: int, number of features to delete
    :param union_weights: dict with keys=models, values=their weights during training
    :param training_percent: float between 0 and 1, percentage of data for training
    :param test_percent: float between 0 and 1, percentage of data for testing
    :param split_job: boolean, True = use multiple cpu cores
    :param result_folder: the folder where the results will be stored (normal text files)
    :return: nothing, after the training and prediction has finished, writes the results into files
    """

    file_prefix = create_prefix(pos_groups, weighing_scale, feature_to_delete, union_weights, training_percent,
                                test_percent)
    process_start = time.time()
    print("Training started: " + file_prefix)
    weight_list = return_best_pos_weight(tagged_sentences, all_labels, pos_groups, weighing_scale, feature_to_delete,
                                         union_weights, training_percent, test_percent, split_job)

    process_end = time.time()
    print("Elapsed time: ", file_prefix, process_end - process_start)

    # Sort accuracy and F1 score
    merge_accuracy = []
    merge_f1 = []
    for element in weight_list:
        merge_accuracy.extend(element)
        merge_f1.extend(element)
    # This is how entries look like
    #  ({'A': 5, 'R': 5, 'V': 5, 'N': 5, 'DEFAULT': 0}, (0.8800877520537714, 0.631544556072858, 0.5874320257269785))
    # we take the second entry in the main tuple and sort by the third value
    merge_f1.sort(reverse=True, key=lambda tup: tup[1][2])
    merge_accuracy.sort(reverse=True, key=lambda tup: tup[1][1])

    number_results = len(merge_accuracy)
    keep_best = 20
    if number_results < keep_best:
        keep_best = number_results

    save_results(result_folder, file_prefix + "_" + "f1_pos_bow.txt", merge_f1[:keep_best])
    save_results(result_folder, file_prefix + "_" + "accuracy_pos_bow.txt", merge_accuracy[:keep_best])


def print_wrong_predictions(docs, prediction, gold_labels, number):
    """
    Function to print wrong predictions of the classifier. Used to analyse results.

    :param docs: The list of tagged sentences
    :param prediction: the class predicted by our model
    :param gold_labels: the true classes
    :param number: The number of wrong predictions to print
    :return: nothing, just print results
    """
    idx_positive = return_wrong_prediction(prediction, gold_labels, number, "positive")
    idx_negative = return_wrong_prediction(prediction, gold_labels, number, "negative")
    idx_neutral = return_wrong_prediction(prediction, gold_labels, number, "neutral")

    def _print_predictions(idx_list, gold_target):
        """
        Internal function, do not use outside
        :param idx_list: list of wrong predictions
        :param gold_target: target label under scrutinity
        :return:
        """

        print("\n#################################Wrong " + gold_target + "#################################")
        for idx in idx_list:
            print("\nPredicted: " + prediction[idx])
            print("\nGold label: " + gold_labels[idx])
            print("\nDoc: ")
            print(docs[idx])
            print("\n#################################")
        print("\n#################################End Wrong " + gold_target + "##########################")

    _print_predictions(idx_positive, "positive")
    _print_predictions(idx_negative, "negative")
    _print_predictions(idx_neutral, "neutral")


def print_best_combination(result, number_to_print=3):
    """
    Automatically parses all result files saved in the results folder and print the best 3 results (accuracy, F1 score)
    :param result: results folder
    :param number_to_print: the number of best results to print
    :return: nothing, just prints results
    """

    best = []
    for x in os.walk(result):
        main_folder = x[0]
        sub_folder = x[1]
        files = x[2]
        for file in files:
            if ".txt" in file:
                try:
                    fp = open(os.path.join(main_folder, file))
                    first_line = [ast.literal_eval(line) for line in fp.readlines()][0][1]
                    best.append((file, first_line,))
                finally:
                    fp.close()

    merge_accuracy = []
    merge_f1 = []
    merge_accuracy.extend(best)
    merge_f1.extend(best)
    merge_f1.sort(reverse=True, key=lambda tup: tup[1][2])
    merge_accuracy.sort(reverse=True, key=lambda tup: tup[1][1])
    print("\nBest 3 accuracies")
    count = 0
    for acc in merge_accuracy:
        file_name = acc[0]
        if "accuracy" in file_name:
            count += 1
            print(acc)
        if count >= number_to_print:
            break
    print("\nBest 3 F1 scores")

    count = 0
    for f1 in merge_f1:
        file_name = f1[0]
        if "f1" in file_name:
            count += 1
            print(f1)
        if count >= number_to_print:
            break


def return_wrong_prediction(prediction, gold_labels, number, target_gold=None):
    """
    Returns the indices of the tweets, which sentiment was predicted wrongly by the classifier

    :param prediction: the class predicted by our model
    :param gold_labels: the true classes
    :param number: The number of wrong predictions to print
    :param target_gold: Optional, allow us to print a number of wrong prediction for a given gold label
    :return: list of indices
    """

    found = 0
    idx_list = []
    size = len(prediction)
    for idx in range(size):
        if prediction[idx] != gold_labels[idx]:
            if (target_gold is not None and target_gold == gold_labels[idx]) \
                    or target_gold is None:
                found += 1
                idx_list.append(idx)
        if found == number:
            break
    return idx_list


def main(argv):
    """
    Main entry point. If the argument "train" is entered on the command line, it will start the training.
    The default behavior is to start the prediction of the saved models
    :param argv: list of command line arguments
    :return:
    """

    # Comment in for the first execution
    # import nltk
    # nltk.download('stopwords')

    # os.getcwd() returns the path until /src
    parent_dir = Path(os.getcwd()).parent.__str__()

    data_set_path = os.path.join(parent_dir, os.path.join("dataset", "processed"))
    model_path = os.path.join(parent_dir, "model")
    results_path = os.path.join(parent_dir, "results")

    # True for train False for predict
    train_or_predict = True

    if len(argv) == 0 or argv[0] != "train":
        train_or_predict = False

    tagged_sentences = os.path.join(data_set_path, 'text_cleaned_pos.csv')
    labels = os.path.join(data_set_path, 'shuffled.csv')

    start_range = 0
    end_range = None  # Set to None to get the whole set...

    tagged_sentences = get_tagged_sentences(data_set_path, tagged_sentences, start_range=start_range,
                                            end_range=end_range, split_pos=False)

    all_labels = get_labels(labels, start_range=start_range, end_range=end_range)
    # pos_groups = {"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}
    # weighing_scale = 5
    # feature_to_delete = 23000
    # union_weights = {'bow': 0.3, 'pos': 0.7, }
    training_percent = 0.7
    test_percent = 0.2
    split_job = True

    number_wrong_predictions_to_print = 20
    model_extension = ".libobj"
    print_best_combination(results_path)

    if train_or_predict:

        prefix_args = [

            # First tests which were run
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 0, {'bow': 0.7, 'pos': 0.3, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 30000, {'bow': 0.7, 'pos': 0.3, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 35000, {'bow': 0.7, 'pos': 0.3, }, training_percent,
            #  test_percent],
            # Test set focusing on having a bigger weight on POS in the Union Feature
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 23000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 30000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 35000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 25000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 28000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 32000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 29500, {'bow': 0.3, 'pos': 0.7, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 30500, {'bow': 0.3, 'pos': 0.7, }, training_percent,
            #  test_percent],

            # # Test like in the paper (POS only, BOW weight = 0)
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 30000, {'bow': 0, 'pos': 1, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 25000, {'bow': 0, 'pos': 1, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 35000, {'bow': 0, 'pos': 1, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 0, {'bow': 0, 'pos': 1, }, training_percent,
            #  test_percent],

            # Test: grouping A and R (POS only, BOW weight = 0)
            # [{"V": ["V"], "N": ["N"], "A+R": ["A", "R"]}, 5, 30000, {'bow': 0, 'pos': 1, }, training_percent,
            # test_percent],
            # [{"V": ["V"], "N": ["N"], "A+R": ["A", "R"]}, 5, 25000, {'bow': 0, 'pos': 1, }, training_percent,
            # test_percent],
            # [{"V": ["V"], "N": ["N"], "A+R": ["A", "R"]}, 5, 35000, {'bow': 0, 'pos': 1, }, training_percent,
            # test_percent],
            # [{"V": ["V"], "N": ["N"], "A+R": ["A", "R"]}, 5, 0, {'bow': 0, 'pos': 1, }, training_percent,
            # test_percent],

            # Test with 50% BOW 50% POS on union
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 0, {'bow': 0.5, 'pos': 0.5, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 25000, {'bow': 0.5, 'pos': 0.5, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 30000, {'bow': 0.5, 'pos': 0.5, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 35000, {'bow': 0.5, 'pos': 0.5, }, training_percent,
            #  test_percent],

            # Patrick?
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 0, {'bow': 0.6, 'pos': 0.4, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 25000, {'bow': 0.6, 'pos': 0.4, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 30000, {'bow': 0.6, 'pos': 0.4, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 35000, {'bow': 0.6, 'pos': 0.4, }, training_percent,
            #  test_percent],

            # Alyona?
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 0, {'bow': 0.8, 'pos': 0.2, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 25000, {'bow': 0.8, 'pos': 0.2, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 30000, {'bow': 0.8, 'pos': 0.2, }, training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 35000, {'bow': 0.8, 'pos': 0.2, }, training_percent,
            #  test_percent],

            # Rafi
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 0, {'bow': 0.5, 'pos': 0.5, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 25000, {'bow': 0.5, 'pos': 0.5, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 30000, {'bow': 0.5, 'pos': 0.5, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 35000, {'bow': 0.5, 'pos': 0.5, },
            #  training_percent,
            #  test_percent],

            # Patrick?
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 0, {'bow': 0.6, 'pos': 0.4, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 25000, {'bow': 0.6, 'pos': 0.4, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 30000, {'bow': 0.6, 'pos': 0.4, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 35000, {'bow': 0.6, 'pos': 0.4, },
            #  training_percent,
            #  test_percent],
            # Alyona?
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 0, {'bow': 0.7, 'pos': 0.3, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 25000, {'bow': 0.7, 'pos': 0.3, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 30000, {'bow': 0.7, 'pos': 0.3, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "R+A": ["R", "A"], "N": ["N"], "E": ["E"]}, 5, 35000, {'bow': 0.7, 'pos': 0.3, },
            #  training_percent,
            #  test_percent],

            # Rafi
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 0, {'bow': 0.5, 'pos': 0.5, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 30000, {'bow': 0.5, 'pos': 0.5, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 35000, {'bow': 0.5, 'pos': 0.5, },
            #  training_percent,
            #  test_percent],

            # Patrick
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 0, {'bow': 0.4, 'pos': 0.6, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 30000, {'bow': 0.4, 'pos': 0.6, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 35000, {'bow': 0.4, 'pos': 0.6, },
            #  training_percent,
            #  test_percent],
            # Patrick
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 0, {'bow': 0.7, 'pos': 0.3, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 30000, {'bow': 0.7, 'pos': 0.3, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 35000, {'bow': 0.7, 'pos': 0.3, },
            #  training_percent,
            #  test_percent],

            # Alyona
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 0, {'bow': 0.6, 'pos': 0.4, },
            #  training_percent,
            #  test_percent],
            # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 30000, {'bow': 0.6, 'pos': 0.4, },
            #  training_percent,
            #  test_percent],
            [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"], "E": ["E"]}, 4, 35000, {'bow': 0.6, 'pos': 0.4, },
             training_percent,
             test_percent],

        ]

        start = time.time()
        print("\nStarted... ")
        for arg in prefix_args:
            data_arg = [tagged_sentences, all_labels]
            data_arg.extend(arg)
            data_arg.append(split_job)
            data_arg.append(results_path)
            run_logic(*data_arg)

        end = time.time()
        print("\nElapsed time overall: ", end - start)
    else:

        print("\nStarting prediction")
        predict_args = [
            # best accuracy
            [{'R': 2, 'V': 4, 'A': 3, 'N': 1}, 29500, {'bow': 0.3, 'pos': 0.7, }],
            # best f1 score. File with 25000 deletion was selected because
            # the Training score was higher, compared to 0 deletions...
            [{'R': 1, 'V': 4, 'A': 1, 'N': 1}, 25000, {'bow': 0.8, 'pos': 0.2, }],

        ]

        for arg in predict_args:
            pos_vocabulary = arg[0]
            pos_group = get_pos_groups_from_vocab(pos_vocabulary)

            train_docs, test_docs, train_labels, test_labels = get_pos_datasets(tagged_sentences, all_labels,
                                                                                pos_group, training_percent,
                                                                                test_percent)
            prefix_arg = []
            prefix_arg.extend(arg)
            prefix_arg.append(training_percent)
            prefix = create_prefix_for_model_persistence(*prefix_arg)

            model_arg = [train_docs, train_labels]
            model_arg.extend(arg)
            serialized_model = os.path.join(model_path, prefix + model_extension)
            model = None
            if not os.path.isfile(serialized_model):
                model = create_fitted_model(*model_arg)
                save_model(model, serialized_model)
            if model is None:
                model = load_model(serialized_model)

            predicted = model.predict(test_docs)
            training_accuracy = model.score(train_docs, train_labels)
            testing_accuracy = model.score(test_docs, test_labels)
            f1 = f1_score(test_labels, predicted, average=None,
                          labels=['neutral', 'positive', 'negative'])
            f1_macro = f1_score(test_labels, predicted, average="macro",
                                labels=['neutral', 'positive', 'negative'])
            print("\nModel: " + prefix, "\nTraining accuracy", training_accuracy, "\nTesting accuracy",
                  testing_accuracy,
                  "\nTesting F1 (neutral, positive, negative)",
                  f1,
                  "\nTesting F1 (macro)",
                  f1_macro, )
            print_wrong_predictions(test_docs, predicted, test_labels, number_wrong_predictions_to_print)
        print("\nEnding prediction")


# Main Entry Point
if __name__ == "__main__":
    main(sys.argv[1:])
