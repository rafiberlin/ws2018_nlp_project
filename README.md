# ws2018_nlp_project


**Project**  
Sentiment Analysis in Twitter Context:  
Incorporating Part-of-speech Tags as Features into a MaxEnt Classifier

**Authors**   
Rafi Abdoule Latif  
Patrick Kahardipraja   
Olena Vyshnevska

**Requirements**   
Python 3.x    
To install the requirements please navigate to the directory of the project in the terminal. Run:  
`pip install -r requirements.txt`  


**File Structure**

├── \_papers  <- scientific literature used for research     
├── baseline  <- implementation of the baseline model: MaxEnt classifier with BoW      
├── dataset  <- the final, canonical data for modeling  
│   ├── raw_data_by_year <- unprocessed data   
│   ├── results  <-  results of feature engineering   
├── features  <- implementation of feature selection technique  
├── model  <- implementation of feature engineering for MaxEnt classifier with POS  
├── process_data  <- implementation of data processing functions  
├── main.py  <-  all functions necessary for training the POS model are called from here  
├── requirements.py  <-  all modules necessary to run scripts  


<!---
- Python 3.x
- https://github.com/clips/pattern (first: "sudo apt-get install gcc", then "conda install -c anaconda mysql-connector-python" finally: "pip install pattern") . Spelling correction
- https://github.com/cbaziotis/ekphrasis (pip install ekphrasis) . Spelling correction 

Versions of Software:

-->

