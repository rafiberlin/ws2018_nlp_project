from pattern.en import suggest
import pandas as pd
import glob
import os
import re


def reduce_lengthening(text):
    """
    Remove any letter that are repeated more than 2 times.
    Source: https://rustyonrampage.github.io/text-mining/2017/11/28/spelling-correction-with-python-and-nltk.html
    :param text:
    :return: words
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def correct_spelling(word):
    """
    Returns the most probable word, if the spelling probability is above 90%
    Example:
    t = correct_spelling("amazzzzziiiiiiing")
    #prints amazing
    print(t)
    :param word:
    :return: corrected word if possible, original word otherwise
    """
    correct_word = word

    attached_punctuation = re.match(r"[?.!;:']$+", word)
    if not attached_punctuation:
        attached_punctuation = ''

    cutoff_prob = 0.9
    word_wlf = reduce_lengthening(word)
    suggested_words = suggest(word_wlf)
    for word, probability in suggested_words:
        if probability >= cutoff_prob:
            correct_word = word
            # to speed up things, this is really slow...
            break
            # cutoff_prob = probability
    return correct_word + attached_punctuation


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
        for fname in all_files:
            with open(fname, "rb") as infile:
                for line in infile:
                    outfile.write(line)


def filter_tab(input, outpath):
    """
    Remove broken lines (which can't be output to proper csv because they contain a tab)
    :param input: complete path to an input file
    :param output: complete path to an output file
    :return:
    """
    df = pd.read_csv(input, index_col=None, sep='\t', header=None, names=['id', 'sentiment', 'text', 'to_delete'],
                     escapechar='“')
    df = df.drop_duplicates(subset="id", keep="first")
    df = df.reset_index(drop=True)
    # workaround somehow, some lines are not being read produced as proper csv
    #  we remove the entries with \t hindering the proper output. we are losing approximatively 500 documents
    patternDel = "\t"
    filt = df['text'].str.contains(patternDel)
    df = df[~filt]
    df.sample(frac=1)
    df.to_csv(outpath + "semval2017_task4_subtask_a_shuffled.csv", header=None, encoding="utf-8", escapechar='“',
              columns=['sentiment', 'text'],
              index=True)
    df.to_csv(outpath + "semval2017_task4_subtask_a_text_only.csv", header=None, encoding="utf-8", escapechar='“',
              columns=['text'], index=False)


def clean_data(input, output):
    """
    The input file will be cleaned as and out to output file as follow=:
    - replace username by token @GENERICUSER
    - replace email by token EMAIL@GENERIC.COM
    - replace urls by http://genericurl.com
    - spelling correction

    :param input:
    :param output:
    :return:
    """
    with open(output, mode="w", encoding="utf-8") as outfile:
        with open(input, mode="r", encoding="utf-8") as infile:
            for line in infile:
                # replace email addresses by generic email token
                # source: https://emailregex.com/
                line = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "EMAIL@GENERIC.COM", line)
                # replace user name by a generic user name token
                # source https://stackoverflow.com/questions/2304632/regex-for-twitter-username
                line = re.sub(r"@[A-Za-z0-9_]{1,15}", "@GENERICUSER", line)
                # replace URL by a generic URL token
                # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
                line = re.sub(
                    r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*',
                    "http://genericurl.com", line)
                cleaned_line = ' '.join([correct_spelling(word) for word in line.split()])
                outfile.write(cleaned_line + "\n")


def create_files_for_analysis(path):
    print("Start")
    merge_files_as_binary(path, path + "semval2017_task4_subtask_a_all_raw.csv")
    filter_tab(path + "semval2017_task4_subtask_a_all_raw.csv", path)
    print("Cleaning")
    clean_data(path + "semval2017_task4_subtask_a_text_only.csv",
               path + "semval2017_task4_subtask_a_text_cleaned.csv")
    print("Finish")


MAIN_PATH = "F:\\UniPotdsam\\WS2018\\Subtask_A\\"

# clean_data still buggy.
create_files_for_analysis(MAIN_PATH)
