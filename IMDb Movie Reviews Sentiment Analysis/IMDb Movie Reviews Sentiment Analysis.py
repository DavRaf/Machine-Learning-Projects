
# coding: utf-8

# In[2]:

import numpy as np

# load the data
reviews_train = []
for line in open('movie_data/full_train.txt', 'r', encoding="utf8"):
    reviews_train.append(line.strip())
    
reviews_test = []
for line in open('movie_data/full_test.txt', 'r', encoding="utf8"):
    reviews_test.append(line.strip())


# In[5]:


# preprocess data
import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)


# In[23]:


from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(' '.join([word for word in review.split()
                                           if word not in english_stop_words]))
    return removed_stop_words

removed_stop_words = remove_stop_words(reviews_train_clean)


# In[25]:


from nltk.stem.porter import PorterStemmer

def get_stemmed_text(corpus):
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

stemmed_reviews = get_stemmed_text(reviews_train_clean)


# In[27]:


from nltk.stem import WordNetLemmatizer

def get_lemmatized_text(corpus):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

lemmatized_reviews = get_lemmatized_text(reviews_train_clean)


# In[36]:


# vectorize data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

n_gram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
n_gram_vectorizer.fit(reviews_train_clean)
X = n_gram_vectorizer.transform(reviews_train_clean)
X_test = n_gram_vectorizer.transform(reviews_test_clean)

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    svm = LinearSVC(C=c)
    lr.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    print("Linear Regression Accuracy for C={}: {}".format(c, accuracy_score(y_val, lr.predict(X_val))))
    print("SVM Accuracy for C={}: {}".format(c, accuracy_score(y_val, svm.predict(X_val))))


# In[34]:


final_lr = LogisticRegression(C=0.5)
final_svm = LinearSVC(C=0.01)
final_lr.fit(X, target)
final_svm.fit(X, target)
print("Linear Regression Accuracy: {}".format(accuracy_score(target, final_lr.predict(X_test))))
print("SVM Accuracy: {}".format(accuracy_score(target, final_svm.predict(X_test))))

import pickle
import os

dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(english_stop_words,
            open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
            protocol=4)
pickle.dump(final_lr, 
            open(os.path.join(dest, 'classifier.pkl'), 'wb'),
            protocol=4)
