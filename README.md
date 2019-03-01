# Sentiment Analysis in Twitter Context: <br/> Incorporating Part-of-Speech Tags as Features <br/> into a MaxEnt Classifier

The project investigates whether including Part-of-Speech (POS) information can help improve Sentiment Analysis of Twitter data. We attempt to maximize the performance of a Maximum Entropy (MaxEnt) Classifier that categorises tweets into three categories with regards to their sentiment: positive, negative and neutral.

### Motivation 
Latent features might carry some important information helpful with the classification task. We focus on one type of latent feature: POS category of a word. 

* By learning appropriate weighting schemes for POS categories our intention is to reinforce the strength of more discriminative categories 

* Distinguishing between polysemous words with the help of POS can help classification


### Problem Statement
If including POS categories is helpful:
* Which categories are strong indicators of sentiment? How significant are the differences?
* What are the optimal weighting schemes? 
* What linguistic phenomena is not captured by POS categories?


### Inspiration
Our approach is based on:   
Chris Nicholls and Fei Song. *Improving Sentiment Analysis with Part-of-Speech Weighting*. Volume 3, pages 1592 – 1597, 08 2009. doi: 10.1109/ICMLC.

We implement a Feature Selection Technique from:   
Jun Yan, Ning Liu, Benyu Zhang, Shuicheng Yan, Zheng Chen, Qiansheng Cheng, Weiguo Fan, and Wei-Ying Ma. OCFS: Optimal Orthogonal Centroid Feature Selection for Text Categorization. In *28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 122–129. Association for Computing Machinery, Inc., January 2005.

The idea of merging POS categories together for calculation came from:   
Rui Xia and Chengqing Zong. A POS-based Ensemble Model for Cross-domain Sentiment Classification. In *Proceedings of 5th International Joint Conference on Natural Language Processing* page 616. Asian Federation of Natural Language Processing, in 2011

## Getting Started

### Prerequisites
Python 3.x

### Installing
To install the requirements please navigate to the directory of the project in the terminal. Run:  
`pip install -r requirements.txt`

## Running the Code  

To start the project, open a terminal/command window, navigate to the src/ folder and enter: 

```
python main.py [train] [devset] [reshuffled] [no_shuffle] [baseline] 

# Optional Arguments:

  train                     # trains models specified in main.py. Saves 20 best weighting combinations in results directory
  devset                    # train on development set
  reshuffled                # use processed, reshuffled data for training and testing
  no_shuffle		    # use processed, equally sized, reshuffled data for training and testing
  baseline                  # output classification reports for bow and tfidf models

```

Running `main.py` without arguments will load our best-performing model and start sentiment class prediction on test data.   
Output of `main.py` without arguments:
* A list of ten best-performing models according to accuracy metric
* A list of ten best-performing models according to F1 metric
* Training accuracy, testing accuracy
* Classification report for the best-performing model    
By default data from equally sized shuffled data from processed/equal_classes_reshuffled directory is used. 

### File Structure

    ├── \_papers                                # scientific literature used for research
    ├── dataset                                 # data used for training, development and testing
    │   ├── processed                           # processed data
    |   ├── processed_reshuffled                # processed data, with documents shuffled
    |   ├── processed_equal_classes_reshuffled  # processed data, with equal number of docs per sentiment class, reshuffled
    │   ├── raw                                 # unprocessed data   
    ├── models                                  # trained models, can be loaded for prediction through main.py  
    ├── results                                 # prediction results of trained models
    │   ├── best                                # top 10 results
    │   ├── all                                 # all resutls
    │   |   ├── data_                           # trained and tested on data in dataset/processed
    │   |   ├── data_reshuffled                 # trained and tested on data in dataset/processed_reshuffled
    │   |   ├── data_equal_classes_reshuffled   # trained and tested on data in dataset/processed_equal_classes_reshuffled
    │   ├── visualization                       # heat maps of baseline and bow+pos results
    ├── src                                     # project source code  
    │   ├── baseline                            # implementation of baseline models: MaxEnt classifier with BoW and TfIdf    
    │   ├── features                            # implementation of feature selection technique and POS Vectorizer
    │   ├── model                               # implementation of transformers and methods for training MaxEnt models
    │   ├── data                                # implementation of methods to process raw data
    │   ├── main.py                             # main script to train models and/or predict class labels
    └── requirements.txt                        # modules necessary to run scripts

### Data

We use data from a shared task on Sentiment Analysis on Twitter, which is part of the International Workshop on Semantic Evaluation (SemEval).

Sara Rosenthal, Noura Farra, and Preslav Nakov. SemEval-2017 task 4: Sentiment analysis in Twitter. In _Proceedings of the 11th International Workshop on Semantic Evaluation_, SemEval ’17, Vancouver, Canada, August 2017. Association for Computational Linguistics.

Specifically, we use the annotated data in English from Subtask A: ”Message Polarity Classification”, which consists of tweets labelled as ”positive”, ”negative” and ”neutral”.

### Data Processing
1. The dataset consists of approximately 60 000 tweets divided into 12 files. We combined all raw data into one file `all_raw` under dataset/raw. 
2. Data Cleaning  
    * Fixed unicode encoding issue
    * Usernames are replaced by token @GENERICUSER  
    * Emails are replaced by token EMAIL@GENERIC.COM   
    * URLs are replaced by http://genericurl.com   
    * Numbers are replaced by GENERICNUMBER   
    * Letters that are repeated more than 2 times within a word (e.g. "paaaaarty") are removed   
    * Quotes, tabs, other unwanted characters are removed   
    * HTML tags are removed
3. ARK Tagger is used for tagging sentences with Part-of-Speech categories. The Tagger is optimized for Twitter data.

Remark: One of our approach consisted in removing the class distribution imbalance. This resulted in an overall smaller dataset of nearly 40000 documents.

Olutobi Owoputi, Brendan O’Connor, Chris Dyer, Kevin Gimpel, Nathan Schneider, and Noah A. Smith. Improved Part-of-Speech Tagging for Online Conversational Text with Word Clusters. In _Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_, pages 380–390. Association for Computational Linguistics, 2013. URL http://aclweb.org/anthology/N13-1039.


To tag the data navigate to the directory where `the runTagger.sh` is located, then run the following command in the terminal:

`./runTagger.sh --output-format conll <name-of-file-to-tag>`


## Results

### Findings: 
Adding POS categories as features does improve the performance of a Bag-of-Words classifier.   
However, TFIDF classifier scores overall the highest.

|          | BOW    | TFIDF  | BOW+POS | TFIDF+POS | BOW+TFIDF+POS |
|----------|--------|--------|-------|-----------|---------------|
| Accuracy | 0.6374 | 0.6515 | 0.6457  | 0.6425    | 0.6440        |
| F1 macro | 0.6383 | **0.6512** | *0.6467*  | 0.6428    | 0.6448        |

**Best Bag of Words + Part-of-Speech Model:**    

Weighting Scheme:     
Adjectives:     4   
Interjections:  4   
Emoticons:      2   
Adverbs:        1   
All Other POS:  0

Features to delete (OCFS): 35000    
Model Weights: BOW 50%, POS 50%    


    === Classification Report for BOW POS (Test Data) ===
    
        Testing Accuracy:  0.6457698447250036 

                  precision  recall     f1-score     support

        negative  0.68950373 0.67847882 0.68394685      2314
         neutral  0.56036636 0.58142549 0.57070172      2315
        positive  0.69349005 0.67816092 0.68573983      2262

       micro avg  0.64576984 0.64576984 0.64576984      6891
       macro avg  0.64778672 0.64602174 0.64679613      6891
    weighted avg  0.64742915 0.64576984 0.64649122      6891

![bow_pos 4_a4_default0_e2_r1_35000_bow_0 5_pos_0 5_0 7](https://user-images.githubusercontent.com/25862134/53637384-4b846300-3c23-11e9-8c73-c99af40c6820.png)




**Baseline: BoW**

    === Classification Report for BOW (Test Data) ===

	  Testing Accuracy:  0.637498186039762 

                  precision  recall     f1-score     support

        negative  0.68521739 0.68107174 0.68313827      2314
         neutral  0.55042017 0.56587473 0.55804047      2315
        positive  0.68159204 0.66622458 0.67382070      2262

       micro avg  0.63749819 0.63749819 0.63749819      6891
       macro avg  0.63907653 0.63772368 0.63833315      6891
    weighted avg  0.63874284 0.63749819 0.63805370      6891


![bow](https://user-images.githubusercontent.com/25862134/53638170-bafb5200-3c25-11e9-9601-a516d04719ad.png)


**Baseline: TfIdf**

    === Classification Report for TFIDF (Test Data) ===

	  Testing Accuracy:  0.6515745174865767 

                  precision  recall    f1-score      support

        negative  0.68898305 0.70267934 0.69576380      2314
         neutral  0.57932264 0.56155508 0.57030050      2315
        positive  0.68386533 0.69142352 0.68762365      2262

       micro avg  0.65157452 0.65157452 0.65157452      6891
       macro avg  0.65072367 0.65188598 0.65122932      6891
    weighted avg  0.65046322 0.65157452 0.65094294      6891

![tfidf](https://user-images.githubusercontent.com/25862134/53638236-f269fe80-3c25-11e9-982e-5e0408f1554c.png)


## Authors
Rafi Abdoule Latif  
Patrick Kahardipraja   
Olena Vyshnevska


## Acknowledgments

The Project has been developed as part of the *Advanced Natural Language Processing* course at the University of Potsdam.   
Winter Semester 2018/2019

<!---
- Python 3.x
- https://github.com/clips/pattern (first: "sudo apt-get install gcc", then "conda install -c anaconda mysql-connector-python" finally: "pip install pattern") . Spelling correction
- https://github.com/cbaziotis/ekphrasis (pip install ekphrasis) . Spelling correction 
- Sometimes, the tkinter module is not installed automatically. Use 'sudo apt-get install python3-tk'

-->

