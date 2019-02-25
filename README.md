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

**How to use**
To start the project, open a terminal /command window, navigate to the src/ folder and enter : `python main.py`
This will start the project in the prediction mode, where the saved models will output their score in the console.

Starting the project with `python main.py train` will start the project in training mode; it will save the best POS 
weighing combinations for a list of POS grouping saved in the main.py script. (see prefix_args variable to edit this list)

**File Structure**

├── \_papers  <- scientific literature used for research     
├── dataset  <- the final, canonical data for modeling  
│   ├── processed <- processed data   
│   ├── raw  <-  unprocessed data   
├── model  <- location for trained models  
├── results <-  results of feature engineering  
├── src  <-  contains the project source code  
│   ├── baseline  <- implementation of the baseline model: MaxEnt classifier with BoW      
│   ├── features  <- implementation of feature selection technique  
│   ├── model  <- implementation of feature engineering for MaxEnt classifier with POS  
│   ├── data  <- implementation of data processing functions  
│   ├── main.py  <-  all functions necessary for training the POS model are called from here  
├── requirements.txt  <-  all modules necessary to run scripts  


<!---
- Python 3.x
- https://github.com/clips/pattern (first: "sudo apt-get install gcc", then "conda install -c anaconda mysql-connector-python" finally: "pip install pattern") . Spelling correction
- https://github.com/cbaziotis/ekphrasis (pip install ekphrasis) . Spelling correction 

Versions of Software:

-->

