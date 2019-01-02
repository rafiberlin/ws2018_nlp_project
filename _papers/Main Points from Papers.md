#Weighting Schemes: Improving Sentiment Analysis with Part-Of-Speech Weighting
1. How to weigh PoS?
- The paper specify a strength for each PoS. Here 4 classes of PoS is used (noun, verb, adjective, verb) with each strength ranging from 1-5. Then the feature weight is normalized over all PoS strengths in the document
2. How to assign initial strength?
- As there are 4 classes, there are 625 possible combinations (5^4). Here we can try for every possible combination, and use OCFS to reduce our feature. The paper also include Improved Iterative Scaling to fine tune the results, but I think this can be optional for us, as we aim to investigate different PoS weighing scheme. Based on the feature that passed our OCFS filter and result on testdata, we can discuss which PoS plays an important role to capture the sentiment.

<!--
RL: If I understood it correctly:

For example: if we have the 2 following documents:
 I-N love-VB cute-ADJ cats-N : Positive
 I-N hate-VB ugly-ADJ dogs-N : Negative

In this example, we have only 5^3 combinations; but we would basically train 5^3 times with the corresponding starting weights (1st classifier ADJ=5, N=4, V=3, ... 125 classifier Adj=1 , N=2, V=5 )  ) and look which one of the starting weights delivered the better results?
We might run into a computing problem if we decide to compute all combinations. Neglecting the testing set with 500 docs, if we split the main dataset (1.6 Million) into
60% Training, 20%Dev, 20%Test => 20% Dev gives 320000 docs, which will make a huge computational difference... In the article, they only use 174 documents to find out the best combination.
Maybe it s not a bad idea to use the dev set with only 500 documents.
-->


#OCFS_optimal_orthogonal_centroid_feature_selection_for_text_categorization
1. How to use OCFS to reduce the number of features: See Chapter "4.3 An Illustrating Example".
Looks like "simple" mean calculation with cutoff values for selection...


#A POS-based Ensemble Model for Cross-domain Sentiment Classification
1.  This paper concentrates on Transfer Learning using POS Tag Groups for sentiment classification and Ensemble methods to combine different learning results into algorithm for prediction.
2. In the paper, they learn the weights of each group individually unlike the paper *Weighting Schemes: Improving Sentiment Analysis with Part-Of-Speech Weighting*.
The individual results are combined with this ensemble approach to build a better prediction.
<!--RL: While the cross domain part of this paper is not so interesting in itself, the grouping problematic is interesting for our project(which POS are significant for classification in a general way? how to group them?)
Maybe we could use the grouping from this paper ( J for [adjectives, adverbs] V for [verbs],  N for [nouns], [O the other POS tags] ) with the methodology used in Weighting Schemes: Improving Sentiment Analysis with Part-Of-Speech Weighting
Or even better, only use J for [adjectives, adverbs] V for [verbs],  N for [nouns] to reduce the search of the best combination in Weighting Schemes: Improving Sentiment Analysis with Part-Of-Speech Weighting
-->



#Twitter Part-of-Speech Tagging for All: Overcoming Sparse and Noisy Data
1. This paper points to the best POS Tagger for twitter ( slang handling for example). <!--RL: What I am not sure of => if we need to perform spelling correction before or if the Tagger will be to do that on the fly to assign the right POS-->

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
