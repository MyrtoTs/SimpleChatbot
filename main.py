import nltk
from sklearn.feature_extraction.text import CountVectorizer

import numpy
import tflearn
import tensorflow
import random

import json

with open('intents.json') as file:
    data = json.load(file)

# initializations
words, labels, docs_x, docs_y = [], [], [], []

for intent in data['intents']:
    for pattern in intent['patterns']:
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

print(f"\nDoc_x now are: \n {docs_x} \n\n while docs_y are: \n {docs_y} \n\n and labels are: \n {labels}")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs_x)
bag_of_words = X.toarray()

Y = vectorizer.fit_transform(docs_y)
bag_of_words_output = Y.toarray()

with open('BOWvectorizer.txt', 'w') as f:
    f.write(f"bag of words training array of type {type(bag_of_words)} of size {bag_of_words.shape} : \n {bag_of_words}\n")
    f.write(f"bag of words output array of size {type(bag_of_words_output)}: \n {bag_of_words_output}\n")