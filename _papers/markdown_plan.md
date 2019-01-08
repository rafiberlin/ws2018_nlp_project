*step one: Baseline*:
- 1. Dataset for project: Sentiment Analysis 140
   *(Twitter messages divided in 3 categories (positive, negative, neutral)dataset developed by Stanford University Students during NLP Lectures)*

- 2. Pre-processing of data
    - spelling correction *(basic method but better than nothing. pattern.en API will provide an easy way to clean data. word will be corrected if the result is > 0.9)*
    - replacing username and url by equivalence token to reduce space *(see main points from Twitter Sentiment Classification using Distant Supervision)*
    - heldout dataset: dataset is big enough (1.6 Millions Training documents + 500 Test)=> 70% training, 10% for dev and 20% (+ 500)for testing 
    - check if repeated tweets: if so, delete repeats
    - Foreign languages in data: leave as it is at first. If the foreign language words affect the result too much:
try to determine if text is english with https://stackoverflow.com/questions/43377265/determine-if-text-is-in-english

    
- 3. Implement 2 MaxEnt baseline classifiers (on pre-processed data): 
    1. With Unigram (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    2. With TD-IDF (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). TF-IDF scores are initial weights.

Why TD-IDF? in "Word clustering based on POS feature for efficient twitter sentiment analysis" introduction => described as one of the most robust and efficient schemes for feature weighing.)
    
*step two: PoS Tagging*:

- Two possible taggers. Check online for comparisons first. If not: test which one is faster and use it for projects: 
    1. GATE (see "Twitter Part-of-Speech Tagging for All: Overcoming Sparse and Noisy Data" conclusion, all results in this paper helped to build it. It improved upon the good work already done by the Stanford POS Tagger). This tagger will provide some special handling of slang, which is really helpful for Twitter.
    2. Carnegie Mellon  (http://www.cs.cmu.edu/~ark/TweetNLP/)
	
AV: 4 implementation options: 1) Java program 2) plugin for the language processing framework GATE 3) model for the Stanford tagger, distributed as a single file, for use in existing applications 4) high-speed model that trades about 2% accuracy for doubled pace.
Suggestion: java program? everything else is unfamiliar to me.

- 2. Other data processing step needed => create an input file with one tweet per line.
	The Tagger will output the result in another file. The new file will need another parsing step to get the POS.



*step three: Weighing Schemes*:
- 1. Previous Related Work: summarise what has been done in papers
*From Twitter Part-of-Speech Tagging for All: Overcoming Sparse and Noisy Data => it says that POS alone does not help to improve classification*.
*On the other hand, IMPROVING SENTIMENT ANALYSIS WITH PART-OF-SPEECH WEIGHTING mentions the opposite*
*A POS-based Ensemble Model for Cross-domain Sentiment Classification is more about transfer learning using POS grouping, but the grouping described in the article could be used with the methodology described in IMPROVING SENTIMENT ANALYSIS WITH PART-OF-SPEECH WEIGHTING*
- 2. Choice of weighing schemes to implement
	- IMPROVING SENTIMENT ANALYSIS WITH PART-OF-SPEECH WEIGHTING uses OCFS as feature selection but does it really perform better than TF*IDF? We could first compare train a second classifier identical to baseline but using OCFS to decide, which feature selection strategy to stick with.
AV: Improving SA paper mentions that OCFS was not helpful for the similar weighting scheme. -> So we try TF-IDF.
	- The choice of the dev set size might be important here, as we test every weight combination to get the optimal weights for POS category. 500 docs is generally small, but it will help us here to find the best combination.
	- Which features are used? Unigrams (initial feature weights calculated with word counting), POS Categories (initial feature weight found testing out all combination). The choice of POS Categories further reduces the feature space (words not belonging to the POS group in the paper are not used) => question? do we use hot encoding or count every occurence of the features? Answer: we use binary encoding (1 if feature exists and 0 if it doesn't. Because: The term frequency can be inaccurate (Zipf's Law).)
	- 1 to 5 are initial weights. The classifier then learns the actual weights from different starting points.
	- Iterative step: Once the best weights are found, we can do the same but also using bigram as feature additionaly. not done in the paper but should enhance the results
	- Iterative step: do the same but with the POS grouping used in A POS-based Ensemble Model for Cross-domain Sentiment Classification => Adjective+Adverb in one Group, one group for verbs, one group for nouns

- 3. try out/implement weighing schemes: 
Three classes possible: one class against all point of view.

- 4. record results in a table for easy comparison

*step four: Discussion*:
- 1. describe results
- 2. discuss what results mean, compare weighing schemes
- 3. compare with papers from Related Work section

*step five: Other Classifiers*
- 1. implement the tagging techniques with other classifiers
- 2. Discuss/compare
