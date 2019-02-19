- OCFS:Instead of using a range of feature scores to eliminate, you just have to define a fix number of features to delete. The features with the worst scores will be deleted first to achieve this number
- use the pipeline to be able to perform all combinations. if feature union can work we don't have to use pipeline
- feature union are fed into the pipeline
- at the end, we will to combine the features of BOW with the features of our POS weighing (be it, with merging ADV+ADJ and using Emoticons...)
- itertools.combination can also be naive way to implement all combination
- unified pipeline: union feature BOW + POS Weighing
- there are many parameters we can change to influence the results: 
  - number of features to be deleted, 
  - weighing combinations, 
  - the pos grouping, the l1/l2 regularization of the maxent classifier. Still looking for api in scikit which would save us work for the combination part. And the union feature weights as well
  - k-fold cross validation

  
## What happens in main.py
- get_tagged_sentences: returns a list of documents, where each documents contains a tuple (word,pos)
[[('gas', '^'), ('by', 'P'), ('my', 'D'), ('house', 'N'), ('hit', 'V'), ('$genericnumber', '^'), ('!!', ','), ('i\\u2019m', 'O'), ('going', 'V'), ('to', 'P'), ('chapel', '^'), ('hill', '^'), ('on', 'P'), ('sat', '^'), ('.', ','), (':)', 'E')], [('theo', '^'), ('walcott', '^'), ('is', 'V'), ('still', 'R'), ('shit\\u002c', 'N'), ('watch', 'V'), ('rafa', '^'), ('and', '&'), ('johnny', '^'), ('deal', 'N'), ('with', 'P'), ('him', 'O'), ('on', 'P'), ('saturday', '^'), ('.', ',')], ...]

- get_labels
labels (of sentences):        
    Label
0   positive
1   negative
2   negative
3   negative
4    neutral
5    neutral
6   positive
7   negative
8    neutral
9   negative
10   neutral
11   neutral
