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
    Remove any letter that are repeated more than 2 times.
    Source: https://rustyonrampage.github.io/text-mining/2017/11/28/spelling-correction-with-python-and-nltk.html
    :param text:
    :return: words
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def remove_backslash_carefully(word, last_corrections=None):
    """
    Only remove the starting backslash if all characters coming afterwards
    are word characters (should not destroy emojis)
    :param word:
    :param last_corrections:
    :return:
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
    correct a given word
    :param word:
    :param last_corrections: dictionary to store words where backslash was removed
    :return:
    """
    word = reduce_lengthening(word)
    word = remove_backslash_carefully(word, last_corrections)
    return word


def merge_files_as_binary(path, output, file_pattern="*.txt"):
    """
    given a certain file pattern, merge the conten of all files in the directory "path" to the
    desired ouput file "output"
    :param path: a directory
    :param output: complete path to an output file
    :param file_pattern: type of files to be merged
    :return:
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
    :return:
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
              # quoting=csv.QUOTE_ALL,
              quoting=csv.QUOTE_ALL,
              columns=['sentiment', 'text'],
              index=True)
    escape_char_textonly = " "
    df.to_csv(os.path.join(output_path, "text_only.csv"), header=None, encoding=file_encoding,
              # quoting=csv.QUOTE_ALL,
              quoting=csv.QUOTE_NONE,
              escapechar=escape_char_textonly,
              columns=['text'], index=False)


def clean_data(input_file, output_file):
    """
    The input file will be cleaned as and out to output file as follow=:
    - replace username by token @GENERICUSER
    - replace email by token EMAIL@GENERIC.COM
    - replace urls by http://genericurl.com
    - spelling correction

    :param input_file:
    :param output_file:
    :return:
    """
    spelling_corrections = {}
    # speller = SpellCorrector(corpus="english")

    file_encoding = "utf-8-sig"
    with open(output_file, mode="w", encoding=file_encoding, newline='') as outfile:
        with open(input_file, mode="r", encoding=file_encoding, newline='') as infile:
            for line in infile:
                # replace number
                # source https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch06s11.html
                line = re.sub(r"[0-9]{1,3}(,[0-9]{3})\.[0-9]+", "GENERICNUMBER", line)

                # replace email addresses by generic email token
                # source: https://emailregex.com/
                line = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "EMAIL@GENERIC.COM", line)
                # replace user name by a generic user name token
                # source https://stackoverflow.com/questions/2304632/regex-for-twitter-username
                line = re.sub(r"@[A-Za-z0-9_]{1,15}", "@GENERICUSER", line)
                # replace URL by a generic URL token
                # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
                line = re.sub(
                    r'http\S+',
                    # r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*',
                    "http://genericurl.com", line)
                # replace any formatted / unformatted numbers and replaces it
                line = re.sub(
                    r'\b\d[\d,.]*\b',
                    "GENERICNUMBER", line)
                # Remove : and - surrounded by characters
                line = re.sub('\s[:-]\s', " ", line)

                # cleaned_line = ' '.join([correct_spelling(word, spelling_corrections, True) for word in line.split()])
                # cleaned_line = ' '.join([correct_spelling2(word, spelling_corrections, True,
                #                                           ["@GENERICUSER", "http://genericurl.com",
                #                                            "EMAIL@GENERIC.COM"], speller) for word in line.split()])
                cleaned_line = ' '.join(
                    [apply_word_corrections(word, spelling_corrections).lower() for word in line.split()])
                outfile.write(cleaned_line + "\n")
                # cleaned_line = ' '.join([word for word in line.split()])
                # outfile.write(line + "\n")


def create_files_for_analysis(path, shuffle=False):
    """

    :param path:
    :param shuffle:
    :return:
    """
    print("Start")
    all_raw = os.path.join(path, "all_raw.csv")
    processed_path = os.path.join(Path(__file__).parents[1].__str__(), 'dataset', 'processed')
    merge_files_as_binary(path, all_raw)
    filter_unwanted_characters(all_raw, processed_path, shuffle)
    clean_data(os.path.join(processed_path, "text_only.csv"),
               os.path.join(processed_path, "text_cleaned.csv"))
    print("Finish")


def build_pie_chart(data_frame_labels, chart_title="Label distribution in the SemEval 2017 data set",
                    filename="dataset/label_chart.png"):
    """
    Creates a pie chart of data distribution
    :param data_frame_labels: as returned by process_data.helper.get_labels()
    :param chart_title: The name of the chart
    :param filename: The name of the chart
    :return:
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
    parent_dir = Path(__file__).parents[1]
    train_path = os.path.join(parent_dir.__str__(), 'dataset', 'raw')
    shuffle_data = False
    create_files_for_analysis(train_path, shuffle_data)


if __name__ == "__main__":
    main()
