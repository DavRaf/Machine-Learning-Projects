# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:42:16 2019

@author: David
"""

import pickle
import os
import numpy as np
from vectorizer import vect

dest = os.path.join('movieclassifier', 'pkl_objects')
clf = pickle.load(open(os.path.join(dest, 'classifier.pkl'), 'rb'))
label = {0: 'negative', 1: 'positive'}
example = ['I love this movie']
X_example = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' %(label[clf.predict(X_example)[0]],
                                              np.max(clf.predict_proba(X_example))*100))