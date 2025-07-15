#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:27:06 2025

@author: Serkan Özkan
"""

import numpy as np
from custom_sklearn.decisiontree import CustomDecisionTreeClassifier
from custom_sklearn.neural_network import SimpleNeuralNetwork
from tensorflow.keras.datasets import mnist


# Parameter
input_size = 784  # 28x28 Pixel
hidden_size = 128
output_size = 10
epochs = 10
learning_rate = 0.1

# Laden des MNIST-Datensatzes
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalisierung der Daten
X_train = X_train / 255.0
X_test = X_test / 255.0

# Umwandlung der Bilder in Vektoren
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# np.eye(10) ist eine Funktion aus der NumPy-Bibliothek in Python, die eine Identitätsmatrix der Größe 10×10 erzeugt.
# Eine Identitätsmatrix ist eine quadratische Matrix, bei der alle Elemente der Hauptdiagonale den Wert 1 haben und alle anderen Elemente den Wert 0 haben.
# One-Hot-Encoding der Labels
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# epochs: Eine Epoche ist ein vollständiger Durchlauf durch den gesamten Trainingsdatensatz. 
# Die Anzahl der Epochen gibt an, wie oft das neuronale Netzwerk die gesamten Trainingsdaten sieht. 
# Mehr Epochen können zu einem besseren Training führen, aber es besteht auch die Gefahr von Überanpassung (Overfitting), 
# wenn das Modell zu viele Epochen trainiert wird.

# learning_rate: Dies ist die Lernrate, die bestimmt, wie stark die Gewichte des neuronalen Netzwerks bei jedem Update angepasst werden. 
# Eine zu hohe Lernrate kann dazu führen, dass das Netzwerk nicht konvergiert, während eine zu niedrige Lernrate das Training verlangsamen kann.

# Training
simple_neural_network = SimpleNeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                                            epochs=epochs, learning_rate=learning_rate)
input_weight, output_weight = simple_neural_network.train(X_train, y_train)

custom_classifier = CustomDecisionTreeClassifier()
custom_classifier.fit(input_weight, output_weight)

predictions = custom_classifier.predict(X_test)
accuracy = custom_classifier.accuracy_score(y_test, predictions)

print(f"Predictions: {predictions}")
print(f'Accuracy: {accuracy:.2f}%')
