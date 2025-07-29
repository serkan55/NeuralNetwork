#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 08:45:07 2025

@author: Serkan Özkan
"""

import sys
import os

# Add the parent directory to PYTHONPATH so custom_sklearn can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np

from Chatbot.chatbot import Chatbot
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

# Train neural network
input_size = len(vocabulary)
hidden_size = [64, 64]
output_size = len(set(labels))
epochs = 1000
learning_rate = 0.1

neural_network = SimpleNeuralNetwork(input_size=input_size, hidden_size = hidden_size, output_size=output_size, epochs=epochs, learning_rate=learning_rate)
weights = neural_network.train(X_train, y_train)

predictions = neural_network.predict(X_train)
print(type(predictions))
print(f"Predictions: {predictions}")

# Calculate accuracy
accuracy = neural_network.accuracy_score(y_train, predictions)
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
    prediction = neural_network.predict(bow_message)

    response_options = [tupel[1] for tupel in responses if tupel[0]==prediction]
    print(random.choice(response_options))
