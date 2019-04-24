
# coding: utf-8

# In[7]:


import nltk


# In[ ]:


nltk.download()


# In[4]:


from nltk.tokenize import sent_tokenize, word_tokenize

text = 'Hello students, how are you doing today? The olympics are inspiring, and Python is awesome. You look great today.'
print(sent_tokenize(text))


# In[5]:


print(word_tokenize(text))


# In[6]:


# removing stop words - useless data
from nltk.corpus import stopwords

print(set(stopwords.words('english')))


# In[7]:


example = 'This is some sample text, showing off the stop words filtration'

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

print(word_tokens)
print(filtered_sentence)


# In[8]:


# Stemming words with NLTK
from nltk.stem import PorterStemmer

ps = PorterStemmer()

example_words = ['ride', 'riding', 'rider', 'rides']

for w in example_words:
    print(ps.stem(w))


# In[9]:


# Stemming an entire sentence
new_text = 'When riders are riding their horses, they often think of how cowboys rode horses.'

words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))


# In[ ]:


from nltk.corpus import udhr

print(udhr.raw('English-Latin1'))


# In[1]:


from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')


# In[2]:


print(train_text)


# In[3]:


# Now that we have some text, we can train the PunktSentenceTokenizer

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)


# In[4]:


# Now lets tokenize the sample text

tokenized = custom_sent_tokenizer.tokenize(sample_text)


# In[5]:


print(tokenized)


# In[8]:


# Define a function that will tag each tokenized word with a part of speech
import nltk

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))
        
process_content()


# In[10]:


nltk.help.upenn_tagset()


# In[13]:


# Chunking with NLTK
train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

# Now that we have some text, we can train the PunktSentenceTokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# Now lets tokenize the sample text
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Define a function that will tag each tokenized word with a part of speech
import nltk

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # combine the part of speech tag with a regular expression
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            # print the nltk tree
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)
            
            # draw the chunks with nltk
            #chunked.draw()
            
    except Exception as e:
        print(str(e))
        
process_content()


# In[14]:


# Chunking with NLTK
train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

# Now that we have some text, we can train the PunktSentenceTokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# Now lets tokenize the sample text
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Define a function that will tag each tokenized word with a part of speech
import nltk

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # combine the part of speech tag with a regular expression
            chunkGram = r"""Chunk: {<.*>+}
                                        }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            # print the nltk tree
            print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)
            
            # draw the chunks with nltk
            #chunked.draw()
            
    except Exception as e:
        print(str(e))
        
process_content()


# In[17]:


# Define a function that will tag each tokenized word with a part of speech
import nltk

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=False)
            
            # draw the chunks with nltk
            namedEnt.draw()
            
    except Exception as e:
        print(str(e))
        
process_content()


# In[18]:


import random
import nltk
from nltk.corpus import movie_reviews


# In[20]:


# build a list of documents
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

# shuffle the documents
random.shuffle(documents)

print('Number of documents: {}'.format(len(documents)))
print('First review: {}'.format(documents[0]))

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

print('Most common words: {}'.format(all_words.most_common(15)))
print('The word happy: {}'.format(all_words["happy"]))


# In[21]:


print(len(all_words))


# In[22]:


# We'll use the 4000 most common words as features
word_features = list(all_words.keys())[:4000]


# In[25]:


# Build a find_features function that will determine which pf the 4000 word features are contained in a review
def find_features(document):
    words = set(document)
    features = {}
    
    for w in word_features:
        features[w] = (w in words)
    
    return features

# lets use an example from a negative review
features = find_features(movie_reviews.words('neg/cv000_29416.txt'))
for key, value in features.items():
    if value == True:
        print(key)


# In[26]:


print(features)


# In[27]:


# now lets do it for all the documents
featuresets = [(find_features(rev), category) for (rev, category) in documents]


# In[28]:


# we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

#define a seed for reproducibility
seed = 1

# split the data into train and test sets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state = seed)


# In[29]:


print(len(training))
print(len(testing))


# In[31]:


# how we use sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))


# In[32]:


# train the model on the training data
model.train(training)


# In[33]:


# test on the testing data
accuracy = nltk.classify.accuracy(model, testing)
print('SVC Accuracy: {}'.format(accuracy))

