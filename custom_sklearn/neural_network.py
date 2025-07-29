#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 08:45:07 2025

@author: Serkan Özkan
"""

import numpy as np


class SimpleNeuralNetwork:
    """SimpleNeuralNetwork is a simple feedforward neural network with hidden layer.
    
    It supports training using backpropagation and can be used for classification tasks.
    It includes functions for initializing weights, forward propagation, backward propagation, and training the network.
    The network uses the sigmoid activation function and its derivative for training via backpropagation.
    """    
    def __init__(self, input_size: int, hidden_size: any, output_size: int, epochs: int, learning_rate: float = 0.1):
        """Initializes the SimpleNeuralNetwork with specified parameters.

        Args:
            input_size (int): Number of input features. e.g., 28x28 Pixel = 784
            hidden_size (int/list): Number of neurons in the hidden layer or layers.
            output_size (int): Number of output neurons.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for weight updates.
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_size
        if type(hidden_size) is int:
            self.hidden_sizes = [hidden_size]
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.weights = self._initialize_weights()

    # Definition der Aktivierungsfunktion (Sigmoid)
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Applies the sigmoid activation function to the input.

        Args:
            x (np.ndarray): Input array.
        Returns:
            np.ndarray: Output array after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    # Definition der Ableitung der Aktivierungsfunktion
    @staticmethod
    def _sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the sigmoid function.
        Args:
            x (np.ndarray): Input array. It can be the output of the sigmoid function.
        If x is the output of the sigmoid function, this function will return the derivative.
        Returns:
            np.ndarray: Output array containing the derivative of the sigmoid function.
        """
        return x * (1 - x)

    # np.random.seed(42) wird verwendet, um den Zufallszahlengenerator von NumPy zu initialisieren. 
    # Durch das Setzen eines Seed-Wertes wird sichergestellt, dass die Folge von Zufallszahlen, die von NumPy generiert wird, reproduzierbar ist. 
    # Das bedeutet, dass jedes Mal, wenn Sie den Seed auf denselben Wert setzen und denselben Code ausführen, Sie dieselbe Folge von "Zufallszahlen" erhalten.
    # np.random.randn ist eine Funktion aus der NumPy-Bibliothek in Python, 
    # die verwendet wird, um Zufallszahlen aus einer Standard-Normalverteilung (auch bekannt als Gauß-Verteilung) zu generieren. 
    # Die Standard-Normalverteilung hat einen Mittelwert von 0 und eine Standardabweichung von 1.
    # Initialisierung der Gewichte
    def _initialize_weights(self) -> list:
        """Initializes weights for a simple feedforward neural network.
        The input size, hidden size, and output size are accessed from the class attributes.
        
        Returns:
            tuple: A tuple containing two numpy arrays representing the weights:
                - input_weight: Weights from input layer to hidden layer.
                - output_weight: Weights from hidden layer to output layer.
        """
        np.random.seed(42)
        weights = []
        weights.append(np.random.randn(self.input_size, self.hidden_sizes[0]))
        for i in range(len(self.hidden_sizes) - 1):
            w = np.random.randn(self.hidden_sizes[i], self.hidden_sizes[i + 1])
            weights.append(w)
        weights.append(np.random.randn(self.hidden_sizes[-1], self.output_size))
        return weights

    # np.dot ist eine Funktion aus der NumPy-Bibliothek in Python, die das Skalarprodukt (auch Punktprodukt genannt) von zwei Arrays berechnet.
    # Die genaue Operation, die np.dot durchführt, hängt von den Dimensionen der Eingabe-Arrays ab
    # Vorwärtspropagierung
    def forward_propagation(self, X: np.ndarray) -> tuple:
        """Performs forward propagation through the neural network.

        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).

        Returns:
            tuple: A tuple containing:
                - zs (list of np.ndarray): List of linear combinations (pre-activations) for each layer.
                - activations (list of np.ndarray): List of activations (outputs after applying sigmoid) for each layer.
                  The first element corresponds to the first hidden layer, the last to the output layer.
        """
        activations = [X]
        zs = []
        for weight in self.weights:
            z = np.dot(activations[-1], weight)
            zs.append(z)
            a = self._sigmoid(z)
            activations.append(a)
        del activations[0]
        return zs, activations

    # Rückwärtspropagierung und Gewichtsaktualisierung - LR -> Learning Rate
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, activations: list) -> list:
        """Performs backward propagation and computes the gradients for all weights in the neural network.

        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
            y (np.ndarray): True labels of shape (n_samples, output_size).
            activations (list of np.ndarray): List of activations for each layer from forward propagation.

        Returns:
            list: A list of numpy arrays containing the gradients (weight updates) for each layer,
                  ordered from input to output layer.
        """
        m = X.shape[0]
        weight_updates = []

        # Calculate the error at the output layer
        dZ = activations[-1] - y

        # Iterate backward through the layers
        for i in reversed(range(len(self.weights))):
            if i == 0:
                break
            activation = activations[i - 1]
            # Calculate the weight update for the current layer
            dW = np.dot(activation.T, dZ) / m
            weight_updates.insert(0, dW)

            # Calculate the error for the previous layer
            if i > 0:
                dZ = np.dot(dZ, self.weights[i].T) * self._sigmoid_derivative(activation)

        # Update the weights for the input layer
        dW_input = np.dot(X.T, dZ) / m
        weight_updates.insert(0, dW_input)

        return weight_updates

    # Training des neuronalen Netzwerks
    def train(self, X: np.ndarray, y: np.ndarray) -> list:
        """Trains a simple feedforward neural network using backpropagation.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
            y (np.ndarray): True labels of shape (n_samples, output_size).
        Returns:
            list: The final list of weight matrices after training, ordered from input to output layer.
        """
        for epoch in range(self.epochs):
            zs, activations = self.forward_propagation(X)
            dWs = self._backward_propagation(X, y, activations)
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dWs[i]
                # self.biases[i] -= self.learning_rate * dBs[i]
            if epoch % 100 == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')
        return self.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the class labels for the given input data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).

        Returns:
            np.ndarray: Predicted class labels for each input sample.
        """
        _, activations = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)

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
