def correct_spelling(word, last_corrections=None, to_lower=False,
                     stop_word_list=None):
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
    :param stop_word_list:
    :return:
    """

    if stop_word_list is None:
        stop_word_list = ["@GENERICUSER", "http://genericurl.com", "EMAIL@GENERIC.COM"]

    if word in stop_word_list:
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
        # avoid to repeat the word, if the word does not contain any alphanumeric characters
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

def correct_spelling2(word, last_corrections=None, to_lower=False,
                      stopwords=None,
                      speller=None):
    """
    Based on ekphrasis
    Function harms more than it fixes things...
    :param word:
    :param last_corrections: dictionary to store spelling corrections. Can speed up the processing for huge texts
    :param to_lower:
    :param stopwords:
    :param speller:
    :return:
    """

    if stopwords is None:
        stopwords = ["@GENERICUSER", "http://genericurl.com", "EMAIL@GENERIC.COM", "GENERICNUMBER"],

    if speller is None:
        speller = SpellCorrector(corpus="english")

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
        # avoid to repeat the word, if the word does not contain any alphanumeric characters
        if not punctuation_found or prefix_found.start() != punctuation_found.start():
            guessed_word = guessed_word[prefix_found.end():]
            prefix = prefix_found.group(0)
    guessed_word = speller.correct(guessed_word)
    final_guess = prefix + guessed_word + punctuation
    if last_corrections is not None:
        last_corrections[word] = final_guess
    return final_guess
