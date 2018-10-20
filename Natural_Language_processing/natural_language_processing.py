# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts4

# Cleaning process - We remove all the verbose info from the text data, like numbers, punctuations and also convert stem the words, i.e., convery all words to the same tense()eg - loved to love) to createa  small bag of words

# The re library helps in formatting the text
# review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) - This checks for letteres a-z and A-Z and keeps them in the works and replcaes all the other charactrers with 2nd para(space in this case)
# review.lower() - converts words to lowercase
# nltk - library 
# The nltk library is used to remove the punctuations, verbose words(like 'this')
# nltk has may tools which need to be download before they are used.
# nltk.download(stopwords) - used to download stopwords list for nltk

import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() #
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # The above command will check for all the iccurances of some word in the collection 'word; (it will pick up the words from the list provided in last(which is stopwords))
    # For eg, if we have love, loved or any other tense, it will consider them all as a single word(since they will mean the same for the current dataset,i.e., reviews for a restaurant) This will help in reducing the words that need to be mantained.
    # What words will be kept is managed by stopwords library of words
    review = ' '.join(review)
    corpus.append(review) # collection of reviews with only keywords in specific tense

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
