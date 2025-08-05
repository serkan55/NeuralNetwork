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

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from NeuralNetwork.custom_sklearn.neural_network import SimpleNeuralNetwork

# Beispiel-Anwendung:
if __name__ == "__main__":
    # MNIST laden und auf 8x8 (statt 28x28) verkleinern für Performance
    (train_X, _), (test_X, _) = mnist.load_data()
    train_X = train_X[:1000]  # für Demo kleiner halten
    test_X = test_X[:10]
     
    # # Normalisieren und Umformen der Daten für 28x28
    x_train = train_X.astype('float32') / 255.
    x_test = test_X.astype('float32') / 255.

    hidden_size = [512]
    # hidden_size=[512, 256, 128, 256, 512]
    epochs = 1000
    learning_rate = 0.1
    pixel_size = 28

    # Normalisieren & auf 8x8 runter skalieren
    from skimage.transform import resize
    
    def init_8x8():
        pixel_size = 8
        x_train = np.array([resize(img, (8, 8), anti_aliasing=True) for img in train_X])
        x_test = np.array([resize(img, (8, 8), anti_aliasing=True) for img in test_X])
        hidden_size = [48]
        return pixel_size, x_train, x_test, hidden_size
    # pixel_size, x_train, x_test, hidden_size = init_8x8()

    # Flatten und normalisieren
    X_train = x_train.reshape(len(x_train), -1)
    X_test = x_test.reshape(len(x_test), -1)
    # print("Tatsächliche Labels:", y_test[:10])
    
    # Modell mit Autoencoder initialisieren
    # 8x8 = 64 Pixel
    # 28x28 = 784 Pixel
    input_size = pixel_size * pixel_size
    
    autoencoder = SimpleNeuralNetwork(input_size=input_size,
                                    hidden_size=hidden_size,
                                    epochs=epochs,
                                    learning_rate=learning_rate)
    # Autoencoder trainieren
    reconstructions = autoencoder.train(X=X_train, X_test=X_test)

    # Rekonstruktion auf Testdaten
    reconstructed = autoencoder.reconstruct(X_test)
    
    # --- Vorhersage & Genauigkeit ---
    accuracy = autoencoder.accuracy_score_of_pixel(X_test, reconstructed)
    print("Genauigkeit:", accuracy, "%")

    # Darstellung: Original vs. Rekonstruiert
    fig, axs = plt.subplots(len(reconstructions) +1, 10, figsize=(15, 3))
    for i in range(10):
        axs[0, i].imshow(X_test[i].reshape(pixel_size, pixel_size), cmap="gray")
        axs[0, i].axis("off")
        for index, reconstructed in enumerate(reconstructions):
            index += 1
            axs[index, i].imshow(reconstructed[i].reshape(pixel_size, pixel_size), cmap="gray")
            axs[index, i].axis("off")

    axs[0, 0].set_title("Original")
    for i in range(1, len(reconstructions) +1):
        axs[i, 0].set_title(f"Rekonstruiert Epoch {100 * i}", size=8)
    plt.tight_layout()
    plt.show()
    