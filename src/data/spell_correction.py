import pandas as pd
import glob
import os
import re
import csv
from pathlib import Path
import html
import matplotlib.pyplot as plt


def reduce_lengthening(text):
    """
    Removes any letter that is repeated more than 2 times.
    E.g. "paaaaaarty" => "paarty"
    :param text: text to correct
    :return: corrected text
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def remove_backslash_carefully(word, last_corrections=None):
    """
    Only remove the starting backslash if all characters coming afterwards
    are word characters (should not destroy emojis)
    :param word: word with a backslash
    :param last_corrections: dictionary with already corrected words
    :return: corrected word
    """

    if last_corrections is None:
        last_corrections = {}

    corrected_word = word

    if last_corrections is not None and word in dict.keys(last_corrections):
        return last_corrections[word]

    if word.startswith("\\"):
        backslash_end_regex = r"^\\[\w]+"
        word_only = re.search(backslash_end_regex, word)
        if word_only:
            corrected_word = corrected_word[1:]
            last_corrections[word] = corrected_word
    if word.endswith("\\"):
        backslash_end_regex = r"[\w]+\\$"
        word_only = re.search(backslash_end_regex, word)
        if word_only:
            corrected_word = corrected_word[:-1]
            last_corrections[word] = corrected_word
    return corrected_word


def apply_word_corrections(word, last_corrections=None):
    """
    Reduce lenghtening and remove unnecessary backslashes from input word
    :param word: word to correct
    :param last_corrections: dictionary to store words where backslash was removed
    :return: corrected word
    """
    word = reduce_lengthening(word)
    word = remove_backslash_carefully(word, last_corrections)
    return word


def merge_files_as_binary(path, output, file_pattern="*.txt"):
    """
    Given a certain file pattern, merge the content of all files in the directory "path" to the
    desired output file "output".
    USed to merge individual raw datasets into one all_raw dataset
    :param path: a directory
    :param output: complete path to an output file
    :param file_pattern: type of files to be merged
    :return: nothing, just writes into output file
    """

    all_files = glob.glob(os.path.join(path, file_pattern))
    with open(output, 'wb') as outfile:
        for f_name in all_files:
            with open(f_name, "rb") as infile:
                for line in infile:
                    outfile.write(line)


def filter_unwanted_characters(input_file, output_path, shuffle=False):
    """
    Remove broken lines (which can't be output to proper csv because they contain a tab)
    :param input_file: complete path to an input file
    :param output_path: complete path to an output file
    :param shuffle: reorder the loaded data if True
    :return: nothing, just writes into files
    """
    df = pd.read_csv(input_file, index_col=None, sep='\t', header=None, names=['id', 'sentiment', 'text', 'to_delete'])
    df = df.drop_duplicates(subset="id", keep="first")
    # workaround somehow, some lines are not being read produced as proper csv
    #  we remove the entries with \t hindering the proper output. we are losing approximatively 500 documents

    pattern_deletion = "\t"
    filt = df['text'].str.contains(pattern_deletion)
    df = df[~filt]
    df = df.reset_index(drop=True)
    # Remove quotes and tab
    df['text'] = df.text.str.replace('\'', '')
    df['text'] = df.text.str.replace('\"', '')
    df['text'] = df.text.str.replace('\t', '')
    # html decoding
    df['text'] = df.text.apply(html.unescape)

    if shuffle:
        df = df.sample(frac=1)
        df = df.reset_index(drop=True)
    file_encoding = "utf-8-sig"

    print('filter unwanted', output_path)
    df.to_csv(os.path.join(output_path, "shuffled.csv"), header=None, encoding=file_encoding,
              quoting=csv.QUOTE_ALL,
              columns=['sentiment', 'text'],
              index=True)
    escape_char_textonly = " "
    df.to_csv(os.path.join(output_path, "text_only.csv"), header=None, encoding=file_encoding,
              quoting=csv.QUOTE_NONE,
              escapechar=escape_char_textonly,
              columns=['text'], index=False)


def clean_data(input_file, output_file):
    """
    The input file will be cleaned and teh resutl will be written to a file as follows:
    - replace username by token @GENERICUSER
    - replace email by token EMAIL@GENERIC.COM
    - replace urls by http://genericurl.com
    - spelling correction

    :param input_file: name of input file for cleaning
    :param output_file: name of output file to write results into
    :return: nothing, just writes into a file
    """
    spelling_corrections = {}
    # speller = SpellCorrector(corpus="english")

    file_encoding = "utf-8-sig"
    with open(output_file, mode="w", encoding=file_encoding, newline='') as outfile:
        with open(input_file, mode="r", encoding=file_encoding, newline='') as infile:
            for line in infile:
                # replace number
                line = re.sub(r"[0-9]{1,3}(,[0-9]{3})\.[0-9]+", "GENERICNUMBER", line)

                # replace email addresses by generic email token
                line = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "EMAIL@GENERIC.COM", line)

                # replace user name by a generic user name token
                line = re.sub(r"@[A-Za-z0-9_]{1,15}", "@GENERICUSER", line)

                # replace URL by a generic URL token
                line = re.sub(
                    r'http\S+',
                    "http://genericurl.com", line)

                # replace any formatted / unformatted numbers and replaces it
                line = re.sub(
                    r'\b\d[\d,.]*\b',
                    "GENERICNUMBER", line)

                # Remove : and - surrounded by characters
                line = re.sub('\s[:-]\s', " ", line)

                cleaned_line = ' '.join(
                    [apply_word_corrections(word, spelling_corrections).lower() for word in line.split()])
                outfile.write(cleaned_line + "\n")


def create_files_for_analysis(path, shuffle=False):
    """
    Merges individual raw docs into one dataset file, filters unwanted characters from dataset,
    shuffles it, cleans it with clean_data and writes it into file text_cleaned.csv
    :param path: path to a document with raw tweets
    :param shuffle: if True shuffle the tweets
    :return: nothing, just writes into files
    """
    print("Start")
    all_raw = os.path.join(path, "all_raw.csv")
    processed_path = os.path.join(Path(__file__).parents[2].__str__(), 'dataset', 'processed')
    merge_files_as_binary(path, all_raw)
    filter_unwanted_characters(all_raw, processed_path, shuffle)
    clean_data(os.path.join(processed_path, "text_only.csv"),
               os.path.join(processed_path, "text_cleaned.csv"))
    print("Finish")


def build_pie_chart(data_frame_labels, chart_title="Label distribution in the SemEval 2017 data set",
                    filename="dataset/label_chart.png"):
    """
    Creates a pie chart of data distribution. Used for presentation.
    :param data_frame_labels: sentiment labels for sentences as returned by process_data.helper.get_labels()
    :param chart_title: Title as it appears on the chart itself
    :param filename: The name of file with the chart
    :return: nothing, just saves the chart as .png
    """
    val_counts = data_frame_labels.Label.value_counts()
    label_count = [val_counts["positive"], val_counts["negative"], val_counts["neutral"]]
    # print("count", label_count)
    label = ['positive', 'negative', 'neutral']
    colors = ['lightblue', 'orange', 'lightgray']
    explode = (0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
    plt.pie(label_count, explode=explode, colors=colors, labels=label,
            autopct='%1.1f%%', shadow=True)
    plt.title(chart_title, bbox={'facecolor': '0.95', 'pad': 5})
    plt.savefig(filename)


def main():
    """
    Creates clean data for classification
    :return: Nothing
    """

    parent_dir = Path(__file__).parents[2]
    print(parent_dir)
    train_path = os.path.join(parent_dir.__str__(), 'dataset', 'raw')
    shuffle_data = True
    create_files_for_analysis(train_path, shuffle_data)


if __name__ == "__main__":
    main()
