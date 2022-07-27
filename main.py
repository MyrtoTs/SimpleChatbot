from sklearn.feature_extraction.text import CountVectorizer

import numpy
import tflearn
import tensorflow
import random
import json

with open('intents.json') as file:
    data = json.load(file)

# Data preprocessing
# From file to "lists of strings"
labels, docs_x, docs_y = [], [], []

for intent in data['intents']:
    for pattern in intent['patterns']:
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Lists of strings as input for vectorizers
# from vectorizer we take csr matrix ... -> array -> list
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs_x)
bag_of_words = (X.toarray()).tolist()  # this lol (list of lists) is our dataset.
                                        # each row (list) is a sample

output_vectorizer = CountVectorizer()
Y = output_vectorizer.fit_transform(docs_y)
bag_of_words_output = (Y.toarray()).tolist()

# Neural Network Design
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(bag_of_words[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(bag_of_words_output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Neural Network Training
model.fit(bag_of_words, bag_of_words_output, n_epoch=1000, batch_size=8, show_metric=True)

# Deployment
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")

        if inp.lower() == "quit":
            break
        inpt = vectorizer.transform([inp]).toarray()
        results = model.predict(inpt)

        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
chat()

## don't like results in chat