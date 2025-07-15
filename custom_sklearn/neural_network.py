#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:27:06 2025

@author: Serkan Özkan
"""

import numpy as np


class SimpleNeuralNetwork:
    """SimpleNeuralNetwork is a simple feedforward neural network with hidden layer.
    
    It supports training using backpropagation and can be used for classification tasks.
    It includes functions for initializing weights, forward propagation, backward propagation, and training the network.
    The network uses the sigmoid activation function and its derivative for training via backpropagation.
    """    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, epochs: int, learning_rate: float = 0.1):
        """Initializes the SimpleNeuralNetwork with specified parameters.

        Args:
            input_size (int): Number of input features. e.g., 28x28 Pixel = 784
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for weight updates.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.input_weight = None
        self.output_weight = None

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
    def _initialize_weights(self) -> tuple:
        """Initializes weights for a simple feedforward neural network.
        The input size, hidden size, and output size are accessed from the class attributes.
        
        Returns:
            tuple: A tuple containing two numpy arrays representing the weights:
                - input_weight: Weights from input layer to hidden layer.
                - output_weight: Weights from hidden layer to output layer.
        """
        np.random.seed(42)
        input_weight = np.random.randn(self.input_size, self.hidden_size)
        output_weight = np.random.randn(self.hidden_size, self.output_size)
        return input_weight, output_weight

    # np.dot ist eine Funktion aus der NumPy-Bibliothek in Python, die das Skalarprodukt (auch Punktprodukt genannt) von zwei Arrays berechnet.
    # Die genaue Operation, die np.dot durchführt, hängt von den Dimensionen der Eingabe-Arrays ab
    # Vorwärtspropagierung
    @staticmethod
    def forward_propagation(X: np.ndarray, input_weight: np.ndarray, output_weight: np.ndarray) -> tuple:
        """Performs forward propagation through the neural network.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
            input_weight (np.ndarray): Weights from input layer to hidden layer.
            output_weight (np.ndarray): Weights from hidden layer to output layer.
        Returns:
            tuple: A tuple containing:
                - Z1: Linear combination of inputs and input weights (hidden layer pre-activation).
                - A1: Activated output of the hidden layer (after applying sigmoid).
                - Z2: Linear combination of hidden layer outputs and output weights (output layer pre-activation).
                - A2: Activated output of the output layer (after applying sigmoid).
        """
        Z1 = np.dot(X, input_weight)
        A1 = SimpleNeuralNetwork._sigmoid(Z1)
        Z2 = np.dot(A1, output_weight)
        A2 = SimpleNeuralNetwork._sigmoid(Z2)
        return Z1, A1, Z2, A2

    # Rückwärtspropagierung und Gewichtsaktualisierung - LR -> Learning Rate
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, input_weight: np.ndarray, 
                              output_weight: np.ndarray, A1: np.ndarray, A2: np.ndarray) -> tuple:
        """Performs backward propagation and updates the weights of the neural network.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
            y (np.ndarray): True labels of shape (n_samples, output_size).
            input_weight (np.ndarray): Weights from input layer to hidden layer.
            output_weight (np.ndarray): Weights from hidden layer to output layer.
            A1 (np.ndarray): Activated output of the hidden layer.
            A2 (np.ndarray): Activated output of the output layer.
        Returns:
            tuple: A tuple containing updated weights:
                - input_weight: Updated weights from input layer to hidden layer.
                - output_weight: Updated weights from hidden layer to output layer.
        """
        m = X.shape[0]
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2) / m
        dZ1 = np.dot(dZ2, output_weight.T) * self._sigmoid_derivative(A1) # A1 is the activated output (sigmoid) of the hidden layer
        dW1 = np.dot(X.T, dZ1) / m

        input_weight -= self.learning_rate * dW1
        output_weight -= self.learning_rate * dW2
        return input_weight, output_weight

    # Training des neuronalen Netzwerks
    def train(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Trains a simple feedforward neural network using backpropagation.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
            y (np.ndarray): True labels of shape (n_samples, output_size).
        Returns:
            tuple: A tuple containing the final weights:
                - input_weight: Weights from input layer to hidden layer.
                - output_weight: Weights from hidden layer to output layer.
        """
        input_weight, output_weight = self._initialize_weights()

        for epoch in range(self.epochs):
            Z1, A1, Z2, A2 = SimpleNeuralNetwork.forward_propagation(X, input_weight, output_weight)
            input_weight, output_weight = self._backward_propagation(X, y, input_weight, output_weight, A1, A2)

            if epoch % 100 == 0:
                loss = np.mean((A2 - y) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')

        return input_weight, output_weight
