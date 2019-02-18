from model.pos import *
import time
import nltk


def save_results(data_set_path, filename, results):
    # Save results
    orig_stdout = sys.stdout
    output = os.path.join(data_set_path, filename)
    with open(output, 'w') as file:
        sys.stdout = file
        for item in results:
            print(item)
    sys.stdout = orig_stdout


def create_prefix(p_groups,
                  w_scale,
                  f_to_delete,
                  u_weights,
                  training_percent,
                  test_percent):
    """
    Creates a prefix based on the model parameters
    :param p_groups:
    :param w_scale:
    :param f_to_delete:
    :param u_weights:
    :param training_percent:
    :param test_percent:
    :return:
    """
    prefix_group = "_".join(["-".join(value) for value in p_groups.values()])
    union_weight_prefix = str(u_weights["bow"]) + "_" + str(u_weights["pos"])
    training = str(training_percent) + "_" + str(test_percent)
    prefix = prefix_group + "_" + str(
        w_scale) + "_" + str(f_to_delete) + "_" + union_weight_prefix + "_" + training
    return prefix


def run_logic(tagged_sentences, all_labels, pos_groups, weighing_scale, feature_to_delete,
              union_weights, training_percent, test_percent, split_job):
    """
    Run the main logic of the project given the parameters
    :param tagged_sentences:
    :param all_labels:
    :param pos_groups:
    :param weighing_scale:
    :param feature_to_delete:
    :param union_weights:
    :param training_percent:
    :param test_percent:
    :param split_job:
    :return:
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

    save_results(data_set_path, file_prefix + "_" + "f1_pos_bow.txt", merge_f1[:keep_best])
    save_results(data_set_path, file_prefix + "_" + "accuracy_pos_bow.txt", merge_accuracy[:keep_best])


# Main Entry Point
if __name__ == "__main__":
    # nltk.download('stopwords')
    parent_dir = os.getcwd()
    data_set_path = os.path.join(parent_dir, "dataset")
    tagged_sentences = os.path.join(data_set_path, 'text_cleaned_pos.csv')
    labels = os.path.join(data_set_path, 'shuffled.csv')

    start_range = 0
    end_range = None  # Set to None to get the whole set...
    tagged_sentences = get_tagged_sentences(data_set_path, tagged_sentences, start_range=start_range,
                                            end_range=end_range, split_pos=False)

    all_labels = get_labels(labels, start_range=start_range, end_range=end_range)
    pos_groups = {"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}
    weighing_scale = 5
    feature_to_delete = 23000
    union_weights = {'bow': 0.3, 'pos': 0.7, }
    training_percent = 0.7
    test_percent = 0.2
    split_job = True

    prefix_args = [
        ##Test set focusing on having a bigger weitht on POS in the Union Feature
        # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 23000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
        #  test_percent],
        # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 30000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
        #  test_percent],
        # [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 35000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
        #  test_percent],
        [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 25000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
         test_percent],
        [{"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}, 5, 28000, {'bow': 0.3, 'pos': 0.7, }, training_percent,
         test_percent],
    ]
    start = time.time()
    print("Started... ")
    for arg in prefix_args:
        data_arg = [tagged_sentences, all_labels]
        data_arg.extend(arg)
        data_arg.append(split_job)
        run_logic(*data_arg)

    end = time.time()
    print("Elapsed time overall: ", end - start)
