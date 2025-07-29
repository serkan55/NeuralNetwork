#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:27:06 2025

@author: Serkan Özkan
"""

import numpy as np
from custom_sklearn.neural_network import SimpleNeuralNetwork


class CustomDecisionTreeClassifier:
    """
    CustomDecisionTreeClassifier is a simple decision tree classifier
    based on weighted input and output layers, performing predictions
    via forward propagation.

    Methods:
        fit(input_weight, output_weight): Sets the weights for input and output layers.
        predict(X): Predicts classes for the given input data.
        accuracy_score(y_test, y_pred): Evaluates the accuracy of the model.
    """

    def __init__(self):
        """Initializes the decision tree classifier and sets the weights to None.
        The weights for the input and output layers will be set during training.
        """
        # Initialisierung der Gewichte für die Eingabe- und Ausgabeschichten
        self.input_weight, self.output_weight = None, None
        
    def fit(self, input_weight: np.ndarray, output_weight: np.ndarray) -> None:
        """
        Sets the weights for the input and output layers of the model.

        Args:
            input_weight (np.ndarray): Weight matrix for the input layer.
            output_weight (np.ndarray): Weight matrix for the output layer.
        """
        self.input_weight = input_weight
        self.output_weight = output_weight

    # np.argmax(A2, axis=1) ist eine Funktion aus der NumPy-Bibliothek, die den Index des maximalen Wertes entlang einer bestimmten Achse in einem Array zurückgibt. 
    # Vorhersage der Klassen für die gegebenen Eingabedaten.
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the classes for the given input data.
        This method uses forward propagation to compute the output layer's activations
        and returns the class with the highest activation for each sample.
        
        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted classes, shape (n_samples,).
        """        
        _, _, _, A2 = SimpleNeuralNetwork.forward_propagation(X, self.input_weight, self.output_weight)
        return np.argmax(A2, axis=1)

    def accuracy_score(self, y_test: np.ndarray, y_pred: np.ndarray, ) -> float:
        """Evaluates the accuracy of the model on the test data.
        This method compares the predicted classes with the true classes and calculates
        the accuracy as the percentage of correct predictions.
        
        Args:
            y_test (np.ndarray): True classes, shape (n_samples,).
            y_pred (np.ndarray): Predicted classes, shape (n_samples,).

        Returns:
            float: Accuracy of the model in percent, rounded to two decimals.
        """

        accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
        return round(accuracy * 100, 2)
