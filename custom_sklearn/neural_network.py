#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 08:45:07 2025

@author: Serkan Özkan
"""

from matplotlib import pyplot as plt
import numpy as np


class SimpleNeuralNetwork:
    """A simple feedforward neural network supporting one or more hidden layers.
    This class provides methods for initializing weights, performing forward and backward propagation,
    training the network using backpropagation, making predictions, and evaluating accuracy.
    The network uses the sigmoid activation function for all layers and its derivative for training via backpropagation.
    Attributes:
        input_size (int): Number of input features.
        hidden_sizes (list): List containing the number of neurons in each hidden layer.
        output_size (int): Number of output neurons.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for weight updates.
        weights (list): List of weight matrices for each layer.
    """
    def __init__(self, input_size: int, hidden_size: any, output_size: int = None, epochs: int = 100, learning_rate: float = 0.1):
        """Initializes the SimpleNeuralNetwork with specified parameters.
        Args:
            input_size (int): Number of input features, e.g., 28x28 Pixel = 784.
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
            list: A list containing numpy arrays representing the weights for each layer.
        """
        np.random.seed(42)
        weights = []

        # output_size is none if autoencoder used
        sizes = [self.input_size] + self.hidden_sizes + ([self.input_size] if not self.output_size else [self.output_size])
        for i in range(len(sizes) - 1):
            weights.append(np.random.randn(sizes[i], sizes[i + 1]))
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
        """
        activations = [X]
        zs = []
        for weight in self.weights:
            z = np.dot(activations[-1], weight)
            zs.insert(0, z)
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
            list: A list of numpy arrays containing the gradients (weight updates) for each layer.
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
    def train(self, X: np.ndarray, y: np.ndarray = None, X_test: np.ndarray = None) -> list:
        """Trains a simple feedforward neural network using backpropagation.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
            y (np.ndarray): True labels of shape (n_samples, output_size).
        Returns:
            list: The final list of reconstructions matrices after training.
        """
        reconstructions = []
        # y is none if autoencoder used
        if y is None:
            y = X
        for epoch in range(self.epochs):
            zs, activations = self.forward_propagation(X)
            dWs = self._backward_propagation(X, y, activations)
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dWs[i]
                # self.biases[i] -= self.learning_rate * dBs[i]
            if epoch % 100 == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')

                if X_test is not None:
                    reconstructions.append(self.reconstruct(X_test))
        return reconstructions

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fits the neural network to the training data.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
            y (np.ndarray): True labels of shape (n_samples,).
        """
        # y is none if autoencoder used
        if y is not None and len(y.shape) == 1:
            y = np.eye(self.output_size)[y]
        self.train(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the class labels for the given input data.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
        Returns:
            np.ndarray: Predicted class labels for each input sample.
        """
        _, activations = self.forward_propagation(X)
        # output_size is none if autoencoder used
        # if not self.output_size:
        #     return activations[-1]
        return np.argmax(activations[-1], axis=1)

    def accuracy_score(self, y_test: np.ndarray, y_pred: np.ndarray, ) -> float:
        """Evaluates the accuracy of the model on the test data.
        Args:
            y_test (np.ndarray): True classes, shape (n_samples,).
            y_pred (np.ndarray): Predicted classes, shape (n_samples,).
        Returns:
            float: Accuracy of the model in percent, rounded to two decimals.
        """

        accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
        return round(accuracy * 100, 2)

    def accuracy_score_of_pixel(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluates the accuracy of the model on the pixel level.

        Args:
            y_true (np.ndarray): True pixel values.
            y_pred (np.ndarray): Predicted pixel values.

        Returns:
            float: Accuracy of the model in percent, rounded to two decimals.
        """
        # Ensure the shapes of y_true and y_pred match
        if y_true.shape != y_pred.shape:
            raise ValueError("Shapes of y_true and y_pred do not match.")

        # Round to 0 or 1, since sigmoid is used
        y_true_bin = (y_true > 0.5).astype(int)
        y_pred_bin = (y_pred > 0.5).astype(int)

        # Calculate the number of correct predictions
        correct_pixels = np.sum(y_true_bin == y_pred_bin)

        # Calculate the total number of pixels
        total_pixels = y_true.size

        # Calculate accuracy
        accuracy = correct_pixels / total_pixels

        # Return accuracy as a percentage
        return round(accuracy * 100, 2)
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Performs forward propagation through the autoencoder.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
        Returns:
            np.ndarray: Reconstructed input data.
        """
        return self.forward_propagation(X)[-1][-1]
