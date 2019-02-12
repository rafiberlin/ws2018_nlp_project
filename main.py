from process_data.helper import *
from model.pos import *
#import nltk

# Main Entry Point
if __name__ == "__main__":
    #nltk.download('stopwords')
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
    feature_to_delete = 30
    union_weights = {'bow': 0.7, 'pos': 0.3, }
    training_percent = 0.7
    test_percent = 0.2
    split_job = False

    start = time.time()
    weight_list = return_best_pos_weight(tagged_sentences, all_labels, pos_groups, weighing_scale, feature_to_delete,
                                         union_weights, training_percent, test_percent, split_job)

    end = time.time()
    print("Elapsed time", end - start)

    # Save results
    orig_stdout = sys.stdout
    result_file_name = "pos_bow.txt"
    with open(os.path.join(data_set_path, result_file_name), 'w') as file:
        sys.stdout = file
        for item in weight_list:
            print(item)
    sys.stdout = orig_stdout
