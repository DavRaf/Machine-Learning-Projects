# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:17:51 2019

@author: David
"""

from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

vect = HashingVectorizer(decode_error='ignore',
                         n_features = 1575100,
                         preprocessor=None,
                         tokenizer=preprocess_reviews)