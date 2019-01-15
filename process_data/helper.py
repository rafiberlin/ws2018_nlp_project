from pattern.en import suggest
import pandas as pd
import glob
import os
import re
import csv
from nltk.stem import WordNetLemmatizer
from pathlib import Path


def reduce_lengthening(text):
    """
    Remove any letter that are repeated more than 2 times.
    Source: https://rustyonrampage.github.io/text-mining/2017/11/28/spelling-correction-with-python-and-nltk.html
    :param text:
    :return: words
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def correct_spelling(word, last_corrections=None, to_lower=False,
                     stopwords=["@GENERICUSER", "http://genericurl.com", "EMAIL@GENERIC.COM"]):
    """
    Returns the most probable word, if the spelling probability is above 91%
    Example:
    t = correct_spelling("amazzzzziiiiiiing")
    #prints amazing
    print(t)
    Function harms more than it fixes things...
    :param word:
    :param last_corrections: dictionnary to store spelling corections. Can speed up the processing for huge texts
    :param to_lower:
    :return:
    """

    if word in stopwords:
        return word

    if to_lower:
        word = word.lower()

    if last_corrections is not None and word in dict.keys(last_corrections):
        return last_corrections[word]

    lemmatizer = WordNetLemmatizer()

    cutoff_prob = 0.91
    reduced_word = reduce_lengthening(word)
    guessed_word = reduced_word
    original_guess = guessed_word
    punctuation_regex = r"[\W]+$"
    non_word_prefix_regex = r"^[\W]+"
    punctuation_found = re.search(punctuation_regex, reduced_word)
    prefix_found = re.search(non_word_prefix_regex, reduced_word)

    punctuation = ""
    prefix = ""
    if punctuation_found:
        # assign the word without punctuation
        guessed_word = reduced_word[:punctuation_found.start()]
        punctuation = punctuation_found.group(0)
    if prefix_found:
        # evoid to repeat the word, if the word does not contain any alphanumeric characters
        if not punctuation_found or prefix_found.start() != punctuation_found.start():
            guessed_word = guessed_word[prefix_found.end():]
            prefix = prefix_found.group(0)

    original_guess = guessed_word
    original_lemma = lemmatizer.lemmatize(guessed_word)
    guessed_word = original_lemma
    suggested_words = suggest(guessed_word)
    for suggested_word, probability in suggested_words:

        # dont correct if listed, rollback. Using lemma as work around for automatic plural corrections
        if suggested_word == original_lemma:
            guessed_word = original_guess
            break
        if probability >= cutoff_prob:
            guessed_word = suggested_word
            cutoff_prob = probability

    final_guess = prefix + guessed_word + punctuation
    if last_corrections is not None:
        last_corrections[word] = final_guess
    return final_guess


assert (correct_spelling("#rafi!") == "#rafi!"), "Spelling function wrong"
assert (correct_spelling("gjdksghfvljdslj!!!!!!") == "gjdksghfvljdslj!!"), "Spelling function wrong"
assert (correct_spelling("paaartyyyyy!!!!!!") == "paartyy!!"), "Spelling function wrong"
assert (correct_spelling("Donald") == "Donald"), "Spelling function wrong"
assert (correct_spelling("My!") == "My!"), "Spelling function wrong"
assert (correct_spelling("My") == "My"), "Spelling function wrong"
assert (correct_spelling("#Singer") == "#Singer"), "Spelling function wrong"
assert (correct_spelling("Burbank") == "Burbank"), "Spelling function wrong"
assert (correct_spelling("Dammit") == "Dammit"), "Spelling function wrong"
assert (correct_spelling("--") == "--"), "Test"
assert (correct_spelling("-") == "-"), "Test"
assert (correct_spelling("Musics", None, True) == "musics"), "Spelling function wrong"
# List of known issues. There is also a problem where the speccling function adds non word chracter erandomly...
assert (correct_spelling("#Sims") == "#Aims"), "Spelling function wrong"
assert (correct_spelling("Ari") == "Ri"), "Spelling function wrong"  # Ari like in Ariana Grande...
assert (correct_spelling("NYC", None, True) == "ny"), "Spelling function wrong"
assert (correct_spelling("Don't", None, True) == "dont"), "Spelling function wrong"
assert (correct_spelling("#Trump", None, True) == "#tramp"), "Spelling function wrong"


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
    # workaround somehow, some lines are not being read produced as proper csv
    #  we remove the entries with \t hindering the proper output. we are losing approximatively 500 documents
    patternDel = "\t"
    filt = df['text'].str.contains(patternDel)
    df = df[~filt]
    df = df.reset_index(drop=True)
    df.sample(frac=1, replace=True)
    df = df.reset_index(drop=True)
    file_encoding = "utf-8-sig"
    df.to_csv(outpath + "semval2017_task4_subtask_a_shuffled.csv", header=None, encoding=file_encoding,
              quoting=csv.QUOTE_ALL,
              columns=['sentiment', 'text'],
              index=True)
    df.to_csv(outpath + "semval2017_task4_subtask_a_text_only.csv", header=None, encoding=file_encoding,
              quoting=csv.QUOTE_ALL,
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
    spelling_corrections = {}
    file_enoding = "utf-8-sig"
    with open(output, mode="w", encoding=file_enoding, newline='') as outfile:
        with open(input, mode="r", encoding=file_enoding, newline='') as infile:
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
                # cleaned_line = ' '.join([correct_spelling(word, spelling_corrections, True) for word in line.split()])
                cleaned_line = ' '.join([reduce_lengthening(word).lower() for word in line.split()])
                outfile.write(cleaned_line + "\n")
                # cleaned_line = ' '.join([word for word in line.split()])
                # outfile.write(line + "\n")


def create_files_for_analysis(path):
    print("Start")
    merge_files_as_binary(path, path + "semval2017_task4_subtask_a_all_raw.csv")
    filter_tab(path + "semval2017_task4_subtask_a_all_raw.csv", path)
    print("Cleaning")
    clean_data(path + "semval2017_task4_subtask_a_text_only.csv",
               path + "semval2017_task4_subtask_a_text_cleaned.csv")
    print("Finish")


parent_dir = Path(__file__).parents[1]
MAIN_PATH = os.path.join(parent_dir.__str__(), "dataset/")  # "F:\\UniPotdsam\\WS2018\\Subtask_A_\\"
# clean_data still buggy.
create_files_for_analysis(MAIN_PATH)
