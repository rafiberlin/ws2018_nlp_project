from process_data.helper import *
from baseline.baseline import main as b_main

# Main Entry Point

if __name__ == "__main__":
    DATA_SET_PATH = os.path.join(os.getcwd(), "dataset")
    TAGGED_SENTENCES = os.path.join(DATA_SET_PATH, 'text_cleaned_pos.csv')
    LABELS = os.path.join(DATA_SET_PATH, 'shuffled.csv')

    DEV_RANGE = (0, 6000)
    TEST_RANGE = (DEV_RANGE[1], 18000)
    TRAINING_RANGE = (TEST_RANGE[1], 61212)
    """
    train_docs, train_tags = get_tagged_sentences(folder=DATA_SET_PATH, filename=TAGGED_SENTENCES,
                                                  start_range=TRAINING_RANGE[0],
                                                  end_range=TRAINING_RANGE[1])
    train_labels = get_labels(shuffled_file=LABELS, start_range=TRAINING_RANGE[0],
                              end_range=TRAINING_RANGE[1])
    """

    all_docs, all_tags = get_tagged_sentences(DATA_SET_PATH, TAGGED_SENTENCES)
    all_labels = get_labels(shuffled_file=LABELS)

    # build_pie_chart(all_labels)

    b_main()

    """
    train_docs = extract_range(all_docs, TRAINING_RANGE[0], TRAINING_RANGE[1])
    train_labels = extract_range(all_labels, TRAINING_RANGE[0], TRAINING_RANGE[1])
    train_tags = extract_range(all_tags, TRAINING_RANGE[0], TRAINING_RANGE[1])

    dev_docs = extract_range(all_docs, DEV_RANGE[0], DEV_RANGE[1])
    dev_labels = extract_range(all_labels, DEV_RANGE[0], DEV_RANGE[1])
    dev_tags = extract_range(all_tags, DEV_RANGE[0], DEV_RANGE[1])

    test_docs = extract_range(all_docs, TEST_RANGE[0], TEST_RANGE[1])
    test_labels = extract_range(all_labels, TEST_RANGE[0], TEST_RANGE[1])
    test_tags = extract_range(all_tags, TEST_RANGE[0], TEST_RANGE[1])
    """
