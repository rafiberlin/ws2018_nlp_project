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


## Getting Started

### Prerequisites
Python 3.x

### Installing
To install the requirements please navigate to the directory of the project in the terminal. Run:  
`pip install -r requirements.txt`

## Running the Code

### Prediction
To start the project, open a terminal /command window, navigate to the src/ folder and enter : `python main.py`
This will start the project in the prediction mode, where the saved models will output their score in the console.

### Training 
Starting the project with `python main.py train` will start the project in training mode; it will save the best POS 
weighing combinations for a list of POS grouping saved in the main.py script. (see prefix_args variable to edit this list)

It is also possible to use the development set to train on with both options: `python main.py train devset`  
or `python main.py devset`

### File Structure

    ├── \_papers                                # scientific literature used for research
    ├── dataset                                 # data used for training, development and testing
    │   ├── processed                           # processed data
    |   ├── processed_reshuffled                # processed data, with documents shuffled
    |   ├── processed_equal_classes_reshuffled  # peocessed data, with equal number of docs per sentiment class, reshuffled
    │   ├── raw                                 # unprocessed data   
    ├── models                                  # trained models, can be loaded for prediction through main.py  
    ├── results                                 # results of feature engineering
    │   ├── best                                # top 10 results
    │   ├── all                                 # all resutls
    │   |   ├── data_                           # trained and tested on data in dataset/processed
    │   |   ├── data_reshuffled                 # trained and tested on data in dataset/processed_reshuffled
    │   |   ├── data_equal_classes_reshuffled   # trained and tested on data in dataset/processed_equal_classes_reshuffled
    ├── src                                     # project source code  
    │   ├── baseline                            # implementation of the baseline models: MaxEnt classifier with BoW snd TfIdf    
    │   ├── features                            # implementation of feature selection technique
    │   ├── model                               # implementation of feature engineering for MaxEnt classifier with POS  
    │   ├── data                                # implementation of data processing functions  
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

Olutobi Owoputi, Brendan O’Connor, Chris Dyer, Kevin Gimpel, Nathan Schneider, and Noah A. Smith. Improved Part-of-Speech Tagging for Online Conversational Text with Word Clusters. In _Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_, pages 380–390. Association for Computational Linguistics, 2013. URL http://aclweb.org/anthology/N13-1039.


To tag the data navigate to the directory where `the runTagger.sh` is located, then run the following command in the terminal:

`./runTagger.sh --output-format conll <name-of-file-to-tag>`


## Results

**Baseline: BoW**
![bow](https://user-images.githubusercontent.com/25862134/53578201-7913d280-3b77-11e9-9f38-65be8432d26d.png)

**Baseline: TfIdf**
![tfidf](https://user-images.githubusercontent.com/25862134/53578202-7913d280-3b77-11e9-918c-f237a9706dfa.png)


**Best Result: BoW/TfIdf + PoS**


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

