from pattern.en import suggest
import pandas as pd
import glob
import os
import re
import csv
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import html
import nltk.corpus.reader.conll as conll
from ekphrasis.classes.spellcorrect import SpellCorrector
import matplotlib.pyplot as plt
from nltk.corpus import stopwords as nltk_stopwords


def reduce_lengthening(text):
    """
    Remove any letter that are repeated more than 2 times.
    Source: https://rustyonrampage.github.io/text-mining/2017/11/28/spelling-correction-with-python-and-nltk.html
    :param text:
    :return: words
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def remove_backslash_carefully(word, last_corrections={}):
    """
    Only remove the starting backslash if all characters coming afterwards are word characters (should not destroy emojis)
    :param word:
    :param last_corrections:
    :return:
    """

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


# assert (correct_spelling("#rafi!") == "#rafi!"), "Spelling function wrong"
# assert (correct_spelling("gjdksghfvljdslj!!!!!!") == "gjdksghfvljdslj!!"), "Spelling function wrong"
# assert (correct_spelling("paaartyyyyy!!!!!!") == "paartyy!!"), "Spelling function wrong"
# assert (correct_spelling("Donald") == "Donald"), "Spelling function wrong"
# assert (correct_spelling("My!") == "My!"), "Spelling function wrong"
# assert (correct_spelling("My") == "My"), "Spelling function wrong"
# assert (correct_spelling("#Singer") == "#Singer"), "Spelling function wrong"
# assert (correct_spelling("Burbank") == "Burbank"), "Spelling function wrong"
# assert (correct_spelling("Dammit") == "Dammit"), "Spelling function wrong"
# assert (correct_spelling("--") == "--"), "Test"
# assert (correct_spelling("-") == "-"), "Test"
# assert (correct_spelling("Musics", None, True) == "musics"), "Spelling function wrong"
# # List of known issues. There is also a problem where the speccling function adds non word chracter erandomly...
# assert (correct_spelling("#Sims") == "#Aims"), "Spelling function wrong"
# assert (correct_spelling("Ari") == "Ri"), "Spelling function wrong"  # Ari like in Ariana Grande...
# assert (correct_spelling("NYC", None, True) == "ny"), "Spelling function wrong"
# assert (correct_spelling("Don't", None, True) == "dont"), "Spelling function wrong"
# assert (correct_spelling("#Trump", None, True) == "#tramp"), "Spelling function wrong"


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


def filter_unwanted_characters(input, outpath, shuffle=False):
    """
    Remove broken lines (which can't be output to proper csv because they contain a tab)
    :param input: complete path to an input file
    :param output: complete path to an output file
    :return:
    """
    df = pd.read_csv(input, index_col=None, sep='\t', header=None, names=['id', 'sentiment', 'text', 'to_delete'])
    df = df.drop_duplicates(subset="id", keep="first")
    # workaround somehow, some lines are not being read produced as proper csv
    #  we remove the entries with \t hindering the proper output. we are losing approximatively 500 documents

    patternDel = "\t"
    filt = df['text'].str.contains(patternDel)
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

    df.to_csv(outpath + "shuffled.csv", header=None, encoding=file_encoding,
              # quoting=csv.QUOTE_ALL,
              quoting=csv.QUOTE_ALL,
              columns=['sentiment', 'text'],
              index=True)
    escapechar_textonly = " "
    df.to_csv(outpath + "text_only.csv", header=None, encoding=file_encoding,
              # quoting=csv.QUOTE_ALL,
              quoting=csv.QUOTE_NONE,
              escapechar=escapechar_textonly,
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
    # speller = SpellCorrector(corpus="english")

    file_enoding = "utf-8-sig"
    with open(output, mode="w", encoding=file_enoding, newline='') as outfile:
        with open(input, mode="r", encoding=file_enoding, newline='') as infile:
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


def correct_spelling2(word, last_corrections=None, to_lower=False,
                      stopwords=["@GENERICUSER", "http://genericurl.com", "EMAIL@GENERIC.COM", "GENERICNUMBER"],
                      speller=SpellCorrector(corpus="english")):
    """
    Based on ekphrasis
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

    reduced_word = reduce_lengthening(word)
    guessed_word = reduced_word
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
    guessed_word = speller.correct(guessed_word)

    final_guess = prefix + guessed_word + punctuation
    if last_corrections is not None:
        last_corrections[word] = final_guess
    return final_guess


assert (correct_spelling2("#rafi!") == "#rafi!"), "Spelling function wrong"
assert (correct_spelling2("gjdksghfvljdslj!!!!!!") == "gjdksghfvljdslj!!"), "Spelling function wrong"
assert (correct_spelling2("paaartyyyyy!!!!!!") == "party!!"), "Spelling function wrong"
# List of known issues. There is also a problem where the speccling function adds non word chracter erandomly...
assert (correct_spelling2("#Sims") == "#aims"), "Spelling function wrong"
assert (correct_spelling2("Ari") == "sri"), "Spelling function wrong"  # Ari like in Ariana Grande...
assert (correct_spelling2("NYC", None, True) == "nyc"), "Spelling function wrong"
assert (correct_spelling2("Don't", None, True) == "dont"), "Spelling function wrong"
assert (correct_spelling2("#Trump", None, False) == "#trump"), "Spelling function wrong"


def create_files_for_analysis(path, shuffle=False):
    print("Start")
    merge_files_as_binary(path, path + "all_raw.csv")
    filter_unwanted_characters(path + "all_raw.csv", path)
    # print("Cleaning")
    clean_data(path + "text_only.csv",
               path + "text_cleaned.csv")
    print("Finish")


def extract_range(iterable, start_range=None, end_range=None):
    """
    return a copy of the rows given the start and end range
    :param iterable:
    :param start_range:
    :param end_range:
    :return:
    """
    num_rows = len(iterable)
    if not start_range:
        start_range = 0
    if not end_range:
        end_range = num_rows
    if start_range > end_range:
        start_range = end_range

    return iterable[start_range:end_range]


def get_tagged_sentences(folder, filename, file_extension=".csv", start_range=None, end_range=None, split_pos=True):
    """
    :param folder:     Folder to the tagged sentences
    :param filename: the file to parse
    :param file_extension: ending of the file toi be parsed
    :param split_pos: if false, returns a list of documents, where each documents contains a tuple (word,pos), if true 2 separated lists (one list of words, one list of corresponding pos)
    :return: one or 2 lists, see param split_pos
    """
    corpus = conll.ConllCorpusReader(folder, file_extension, ('words', 'pos'))
    tagged_sentences = extract_range(corpus.tagged_sents(filename), start_range, end_range)
    if not split_pos:
        return tagged_sentences

    sentences_only = []
    tags_only = []

    for tagged_sentence in tagged_sentences:
        words, tags = zip(*tagged_sentence)
        # undo tokenize done by ark tagger adding white space, if needed by scikit
        # sentences_only.append(" ".join(list(words)))
        sentences_only.append(list(words))
        tags_only.append(list(tags))
    return sentences_only, tags_only


def get_labels(shuffled_file, start_range=None, end_range=None):
    """
    used to get encoded labels (negative =0, positive 1, neutral 2) from the /dataset/shuffled.csv file
    :param shuffled_file:
    :param max_rows:
    :return: labels and labels for testing data as pandas.dataframe objects
    """

    df = pd.read_csv(shuffled_file, sep=',', header=None, names=['ID', 'Label', 'Orig'], quoting=csv.QUOTE_ALL,
                     encoding='utf8')
    df = df.drop(['ID', 'Orig'], axis=1)
    labels = df.replace({'Label': {'negative': 0, 'positive': 1, 'neutral': 2}})

    return extract_range(df, start_range, end_range)


def build_pie_chart(data_frame_labels, chart_title="Label distribution in the SemEval 2017 data set",
                    filename="dataset/label_chart.png"):
    """
    Creates a pie chart.
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


def pre_processing(tagged_sentence, pos_grouping=None,
                   default_pos="DEFAULT",
                   stopwords=None, to_lower=True):
    """
    Apply some pre-processing on pre tagged sentences. Stopwords, POS Grouping, lowering case
    :param tagged_sentence:
    :param pos_grouping:
    :param default_pos:
    :param stopwords:
    :param to_lower:
    :return: tagged sentences with removed stopped words and different pos grouping
    """
    if pos_grouping is None:
        pos_grouping = {"V": ["V"], "A": ["A"], "N": ["N"], "R": ["R"]}

    if stopwords is None:
        stopwords = set(nltk_stopwords.words('english'))

    processed_sentences = []
    for sentence in tagged_sentence:
        new_sentence = []
        for word_pos in sentence:
            word = word_pos[0]
            pos = word_pos[1]
            group_found = False
            if to_lower:
                word = word.lower()
            if word in stopwords:
                continue
            for pos_group_key, pos_group_values in pos_grouping.items():
                if pos in pos_group_values:
                    new_sentence.append((word, pos_group_key,))
                    group_found = True
                    break
            # Fallback to avoid empty documents. Exemple tweet that does not contain any of our groups => just emoji + hashtag
            if not group_found:
                new_sentence.append((word, default_pos,))
        processed_sentences.append(new_sentence)
    return processed_sentences


def main():
    parent_dir = Path(__file__).parents[1]
    TRAIN_PATH = os.path.join(parent_dir.__str__(), "dataset/raw_data_by_year/train/")
    shuffle_data = False
    # clean_data still buggy. TODO backslash handling not optimal
    create_files_for_analysis(TRAIN_PATH, shuffle_data)

    TEST_PATH = os.path.join(parent_dir.__str__(), "dataset/raw_data_by_year/test/")
    shuffle_data = False
    # clean_data still buggy. TODO backslash handling not optimal
    create_files_for_analysis(TEST_PATH, shuffle_data)


if __name__ == "__main__":
    main()
