# Author: Mettu Deepika  
## Project 3: The Unredactor  

The unredactor will take redacted documents and return the most likely candidates to fill in the redacted location.  

The project tree struture:  
![image](https://user-images.githubusercontent.com/95551102/167355561-f30e0506-48cd-4a95-8dff-27790cacf31b.png)  

**The command for running the code:**  
*pipenv run python project3.py*  

**The command for running the pytest:**  
*pipenv run python -m pytest*  

**The modules that are used for this project:**  
![image](https://user-images.githubusercontent.com/95551102/167357118-638c4870-495e-47eb-90c0-8b1955f9a96b.png)  

The above modules are installed using the command: pipenv install <module_name>  

The project3.py file contains the 5 following funcctions defined in it:  

- ***text_processing(text):***  
This function is used for cleaning the data. Cleaning here includes, removing digits, punctuations, underscores, and extra whitespaces from the data being read from the link *"https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"*.  

- ***tfidf_dict(document):***  
In this function, we are creating a model using TfidfVectorizer and encoding the cleaned data using tranform. Then, we are storing, feature names and idf values in a dictionary and returning it.  

- ***feature_extract(document,idf_dict):***  
In feature function, we are extracting 5 features.  
1. size of name which is length of name characters  
2. spaces between name entity. If space is not present it will be marked 0  
3. size of sentence, which is number of characters in that sentence  
4. Idf value of left word of redactor, if redactor is the 1st entry in sentence then it will be selected 0. Otherwise it will search for idf value of the left position wordin tfidf dictionary.  
5. Same as above but instead of left position word we are taking right position word. If redactor has the last position then its value will be 0.  


- ***dictvect(data):***  
In this function, we are creating a model using DictVectorizer, encoding the data using transform and converting it to array.

In the main function, all the functions that are defined above are called. We are splitting the given dataset into training and testing data sets with test data of size 0.2. Next, we are using an object of DecisionTreeClassifier() as our prediction model and training it for prediction. Finally, we are printing the F1 score, precsion value and recall values along with the predicted names to the concole.   

**Test Cases:**

The test cases have been defined for all the functions that are defined in project3.py file.

- ***test_test_processing():***  
This is used for testing *test_processing()* function defined in project3.py. This function will assert  True if the data is cleaned correctly and does not contain underscore. Else asserts False.   

- ***test_tfidf_dict():***  
This function is used for testing *tfidf_dict* function defined in project3.py. This function will assert True if the cleaned data and the dictionary with feature names and idf values are returned correctly.  

- ***test_feature_extract():***  
This function is used for testing *feature_extract()* function defined in project3.py. This function will assert True when it returns the features correctly.  


- ***test_dictVect():***  
This function is used for testing *dictvect()* function defined in project3.py. This function will assert True on succesfull creation of DictVectorizer, encosing the data using tranform and converting it to array.  

**Assumptions**:  
- Decision Tree Classifier is the best model (with high accuracy) used for prediction.  

**Possible Bugs:**  
- Pridiction might npt be accurate.  


***References:***  
- https://stackoverflow.com/questions/23464138/downloading-and-accessing-data-from-github-python  
