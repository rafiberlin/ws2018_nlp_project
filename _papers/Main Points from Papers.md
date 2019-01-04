#Weighting Schemes: Improving Sentiment Analysis with Part-Of-Speech Weighting
1. How to weigh PoS?
- The paper specify a strength for each PoS. Here 4 classes of PoS is used (noun, verb, adjective, verb) with each strength ranging from 1-5. Then the feature weight is normalized over all PoS strengths in the document.
AV: filter out all words such, that have PoS tags other than {noun,verb,adjective,adverb} by setting the weights to 0.

2. How to assign initial strength (see chapter 4.3 for details)?
- As there are 4 classes, there are 625 possible combinations (5^4). Here we can try for every possible combination, and use OCFS to reduce our feature. The paper also include Improved Iterative Scaling to fine tune the results, but I think this can be optional for us, as we aim to investigate different PoS weighing scheme. Based on the feature that passed our OCFS filter and result on testdata, we can discuss which PoS plays an important role to capture the sentiment.
3. Initial weights for individual words are assigned used word counting accross document (see chapter 4.1) 
<!--
RL: If I understood it correctly:

For example: if we have the 2 following documents:
 I-N love-VB cute-ADJ cats-N : Positive
 I-N hate-VB ugly-ADJ dogs-N : Negative

In this example, we have only 5^3 combinations; but we would basically train 5^3 times with the corresponding starting weights (1st classifier ADJ=5, N=4, V=3, ... 125 classifier Adj=1 , N=2, V=5 )  ) and look which one of the starting weights delivered the better results?
AV: this is also how I understand it. If there are any words that get PoS tags that are not N,VB,ADJ or ADV - their weights are set to 0.

We might run into a computing problem if we decide to compute all combinations. Neglecting the testing set with 500 docs, if we split the main dataset (1.6 Million) into
60% Training, 20%Dev, 20%Test => 20% Dev gives 320000 docs, which will make a huge computational difference... In the article, they only use 174 documents to find out the best combination.
Maybe it s not a bad idea to use the dev set with only 500 documents.
-->  

AV: 
4. **Why weighting** PoS categories?
    - By doing this, the intention is to get further separation between sentiment carrying words and content words, without completely ignoring entire word categories (such as nouns, which are often content words)

5. **Evaluation** metrics:
    - Precision, recall and accuracy: presented. F measure: used for comparing performance

6. **Baseline** performance:
- baseline classifier uses only the OCFS and the word count feature weighting (BoW)
The way I(AV) understand it is: they compared three baseline classifiers: 1) closed PoS filter with 500 features; 2) closed PoS filter with all features and 3) 1)Stemming, stopword removal and closed PoS filter
- Best baseline performance: BoW + OCFS + Closed POS filter (everything that is not a noun, verb, adjective or adverb is filtered out) + stemming and stop word removal. Feature cutoff: 1000 or no cut-off. F: 72.00%

(?) in Table 2 the average F for baseline: 74.80%, so it increased from initial 72%. It is because of Improved Iterative Scaling? 
(?) Stemming and stopword removal should be easy, if we want to add it to baseline

7. **PoS-enriched classifier performance**: 
    - Best weights: N:2, VB:3, ADV:4, ADJ:5 - 79.20% (baseline: 74.80%)

8. Take away: 
    - adjectives are very strong indicators of sentiment, but adverbs can be just as strong.
    - with similar PoS strength combinations as in paper: using OCFS doesn't have much effect (commonly one or two POS categories that dominate the others in terms of weighting. So an odd low-frequency term that is left in the feature space will contribute little to classification of a document)

#OCFS_optimal_orthogonal_centroid_feature_selection_for_text_categorization
1. How to use OCFS to reduce the number of features that do not contribute to distinguishing between texts: See Chapter "4.3 An Illustrating Example".
Looks like "simple" mean calculation with cutoff values for selection...  
AV:
2. OCFS is aimed at finding features which maximize the distance between the mean feature vectors (centroids) of all classes.
3. How OCFS works:
    4. Computes the centroid vectors for each class
    5. Ranks each feature by its distance from the centroids
    6. Cuts off all features whose distances from the centroids are below a pre-specified threshold


#A POS-based Ensemble Model for Cross-domain Sentiment Classification
1.  This paper concentrates on Transfer Learning using POS Tag Groups for sentiment classification and Ensemble methods to combine different learning results into algorithm for prediction.
2. In the paper, they learn the weights of each group individually unlike the paper *Weighting Schemes: Improving Sentiment Analysis with Part-Of-Speech Weighting*.
The individual results are combined with this ensemble approach to build a better prediction.
<!--RL: While the cross domain part of this paper is not so interesting in itself, the grouping problematic is interesting for our project(which POS are significant for classification in a general way? how to group them?)
Maybe we could use the grouping from this paper ( J for [adjectives, adverbs] V for [verbs],  N for [nouns], [O the other POS tags] ) with the methodology used in Weighting Schemes: Improving Sentiment Analysis with Part-Of-Speech Weighting
Or even better, only use J for [adjectives, adverbs] V for [verbs],  N for [nouns] to reduce the search of the best combination in Weighting Schemes: Improving Sentiment Analysis with Part-Of-Speech Weighting
-->
AV: @RL, we can try your last suggestion. It seems like a small change: the only difference to the grouping in Improving-SA paper is that we combine adverbs and adjective into one group, right? 


3. **Findings**: 
    - across different domains, features with some types of PoS tags are domain-dependent, while some others are domain-free. 
    - When changing domain: part of knowledge changes and other part remains similar. Changes across domains: N>>Uni>V>J>O. The change of N is significantly larger than the other POS types and Uni (unigrams). The distribution of O (words that don't belong to one of 4 main categories) changes the least. Most fetures in J and V are partially domain-free (e.g. “great” and “love” always express a positive meaning in whatever domains.) 

4. **Adjectives**: Even though Improving Sentiment paper suggests that adjectives are the most important indicators of sentiment, "using only adjectives as features actually results in much worse performance than using the same number of most frequent unigrams" (cited: Pang et al., 2002; Benamara et al., 2007)

5. **Learning Weights**: stochastic gradient descent to optimize two criteria: perceptron (Perc) model, and minimal classification error (MCE) criterion. 
=> its sounds like a pretty complicated calculation that they use

6. PoS Tagger: they used an old tagger from 1997. It seems pretty outdated to me, which may have affected the results (http://www.inf.ed.ac.uk/resources/nlp/local_doc/MXPOST.html)


#Twitter Part-of-Speech Tagging for All: Overcoming Sparse and Noisy Data
1. This paper points to the best POS Tagger for twitter ( slang handling for example). <!--RL: What I am not sure of => if we need to perform spelling correction before or if the Tagger will be to do that on the fly to assign the right POS--> 
2. 90.5% tagging accuracy

*Twitter Sentiment Classification using Distant Supervision*
All the following points might be relevant for Data Prepocessing and POS Tagging
1. Emoticons could help to understand if a Tweet is positive or negative => not useful for us, the emoticons were already removed.
2. Standard length of twitters => 140 Characters. We should use the length of a document as feature.
Points 3 to 5 see chapter 2.3
3. all usernames (starting by @ followed by a string) replaced by equivalence token USERNAME to reduce feature space. => could be useful as well
4. all urls replaced by equivalence token URL to reduce feature space. => could be useful as well
5. Repeated letters being reduced to max 2 consecutive letters=>ex. huuuuuuuuuungry => huungry. <!--RL: This is used in the spelling correction function I pushed.
see chapter 4.2 Results for following points-->
6. Unigrams + Bigrams performs better than Unigram or Bigrams alone <!--RL: SHould we also use Unigram + Bigram as well in step 3 -->
7. They say that POS alone does not help (no POS grouping used here.) => it will be interesting to compare with our results for MaxEnt



#Twitter Sentiment Classification using Distant Supervision

- **Effect of PoS tagging**: concluded that PoS tags were not useful: performance for MaxEnt increased non-significantly, while performance of Naive Bayes and SVM decreased. 
    - no further informations on weighting of PoS tags. Seems like they used PoS tags just to disambiguate word meaning (e.g. “over” as a verb may have a negative connotation, but as a noun it is neutral, as in "cricket over")
- **Bigrams as Features**: Using only bigrams as features is not useful because the feature space is very sparse. When using both unigram and bigrams as features accuracy improved for Naive Bayes (81.3% from to 82.7%) and Max- Ent (from 80.5 to 82.7). 


# Word clustering based on POS feature for efficient twitter sentiment analysis

- Even though we are not using the weighting scheme from the paper, it is a very recent paper (2018) that contains explanations for topics relevant to us
- **TF-IDF**: The term frequency and inverse document frequency (TF–IDF)
    - effectively measures the importance of the words among documents: selects the words important for a document and not paying much attention to common words
        - Low value: the word appears evenly in every document
        - High value: frequent only in few documents (also implying that many documents contain this word)
    - the importance of a word depends on 2 things:
        - Term frequency (TF): how often it occurs in the document 
        - inverse document frequency (idf): ratio of total number of documents to num of documents containing the word
    - "greatly increases the accuracy of sentiment analysis"
    - like BoW, size of features is size of vocabulary of the document -> high dimentionality