#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:56:21 2025

@author: Serkan Özkan
"""

import sys
import os
# Add the parent directory to PYTHONPATH so custom_sklearn can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np

from Chatbot.chatbot import Chatbot
from NeuralNetwork.custom_sklearn.decisiontree import CustomDecisionTreeClassifier
from NeuralNetwork.custom_sklearn.neural_network import SimpleNeuralNetwork


# Prepare the chatbot data
chatbot = Chatbot("Chatbot/ChatbotTraining.csv")
data = chatbot.data
labels = chatbot.label_encoder(data.tag)
responses = data.responses.values.tolist()
patterns = data.patterns.values.tolist()
vocabulary = chatbot.get_vocabulary(patterns)

# Prepare training data
X_train = []
for sentence in patterns:
    vec = chatbot.bag_of_words_as_vector(sentence, vocabulary)
    X_train.append(vec)
X_train = np.array(X_train)

y_train = labels
y_train = np.eye(len(set(y_train)))[y_train] # Assuming labels are the indices of the patterns

# print('#' * 50)
# print(f"Len X_train: {len(X_train)}")
# print(f"Len y_train: {len(y_train)}")
# print(f"Len patterns: {len(chatbot.patterns)}")
# print(f"Len vocabulary: {len(chatbot.vocabulary)}")
# print(f"Len labels: {len(chatbot.labels)}")
# print("-" * 50)

# Train neural network
input_size = len(vocabulary)
hidden_size = 64
output_size = len(set(labels))
epochs = 1000
learning_rate = 0.1

nn = SimpleNeuralNetwork(input_size, hidden_size, output_size, epochs, learning_rate)
input_weight, output_weight = nn.train(X_train, y_train)

custom_classifier = CustomDecisionTreeClassifier()
custom_classifier.fit(input_weight, output_weight)

# Vorhersage mit Trainingsdaten (X_train)
predictions = custom_classifier.predict(X_train)
print(f"Predictions: {predictions}")

# Calculate accuracy
accuracy = custom_classifier.accuracy_score(y_train, predictions)
print(f'Accuracy: {accuracy:.2f}%')

responses = list(zip(labels, responses))
print('### Starts ###')
while True:
    message = input('Message: ')
    if message.lower() == 'exit':
        print('Exiting...')
        break
    bow_message = [chatbot.bag_of_words(message, vocabulary)]

    # Predict the response for test dataset
    # find out over bow which category/tag the message belongs to
    prediction = custom_classifier.predict(bow_message)

    response_options = [tupel[1] for tupel in responses if tupel[0]==prediction]
    print(random.choice(response_options))
