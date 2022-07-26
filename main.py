import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

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
        wrds = nltk.word_tokenize(pattern)
        # Are you cash only?
        # Tokenize:['Are', 'you', 'cash', 'only', '?']
        # print(f"Words now are {words} \n while docs_x are {docs_x}\n\n\n")
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# print(f"\nDoc_x now are: \n {docs_x} \n\n while docs_y are: \n {docs_y} \n\n and labels are: \n {labels}")

# Stemming
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
# print(f"\nStemmed words are: \n {words}\n" )

words = sorted(list(set(words)))
    # set => no duplicates
    # list => same type as before
    # sorted => on alphabetical order

# print(f"\n and after sort: \n {words}\n" )

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

bag_of_words = numpy.array(training)
bag_of_words_output = numpy.array(output)

# with open('BOWmanually.txt', 'w') as f:
#     f.write(f"bag of words training array of type {type(bag_of_words)} of size {bag_of_words.shape} : \n {bag_of_words}\n")
#     f.write(f"bag of words output array of size {type(bag_of_words_output)}: \n {bag_of_words_output}\n")

