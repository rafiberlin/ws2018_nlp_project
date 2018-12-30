from pattern.en import suggest
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
    cutoff_prob = 0.9
    word_wlf = reduce_lengthening(word)
    suggested_words = suggest(word_wlf)
    for word, probability in suggested_words:
        if probability >= cutoff_prob:
            correct_word = word
    return correct_word
