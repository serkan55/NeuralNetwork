# NeuralNetwork Project

Dieses Modul bietet eine einfache Möglichkeit, einen Chatbot mit neuronalen Netzen und Entscheidungsbaum-Klassifikatoren zu trainieren und zu testen. Es ist für die Verarbeitung von Textdaten (z.B. Chatbot-Intents) ausgelegt und nutzt eigene Implementierungen von neuronalen Netzen und Entscheidungsbäumen.

---

## Inhaltsverzeichnis

- Features
- Installation
- Dateistruktur
- Voraussetzungen
- Verwendung
  - Daten vorbereiten
  - Training
  - Vorhersage und Genauigkeit
  - Chatbot-Interaktion
- Parameterwahl
- Beispiel
- Lizenz

---

## Features

- Eigene Implementierung eines neuronalen Netzes (`SimpleNeuralNetwork`)
- Eigener Entscheidungsbaum-Klassifikator (`CustomDecisionTreeClassifier`)
- Bag-of-Words-Vektorisierung für Textdaten
- Label-Encoding für Klassifikationsaufgaben
- Training, Vorhersage und Genauigkeitsberechnung
- Interaktiver Chatbot-Modus

---

## Installation

1. **Repository klonen**

   ```bash
   git clone <repo-url>
   cd chatbot
   ```

2. **Abhängigkeiten installieren**

   ```bash
   pip install -r requirements.txt
   ```

   Typische Abhängigkeiten: `numpy`, `pandas`, `nltk`, ggf. eigene Module.

3. **NLTK-Resourcen herunterladen**  
   Im Code werden NLTK-Resourcen wie Stopwords und Tokenizer geladen:
   ```python
   import nltk
   nltk.download("punkt")
   nltk.download("stopwords")
   ```

---

## Dateistruktur

```
NeuralNetwork/
├── custom_sklearn/
│   ├── decisiontree.py
│   └── neural_network.py
│
├── chatbot_training.py
├── custom_sklearn_test.py
│
└── README.md
```

---

## Voraussetzungen

- Python 3.8+
- Die Datei `ChatbotTraining.csv` mit den Spalten:
  - `tag`: Intent-Label (z.B. greeting, goodbye)
  - `patterns`: Beispieltexte
  - `responses`: Mögliche Antworten

---

## Verwendung

### Daten vorbereiten

Die Trainingsdaten werden aus der CSV-Datei geladen und in Bag-of-Words-Vektoren und numerische Labels umgewandelt.

```python
chatbot = Chatbot("Chatbot/ChatbotTraining.csv")
data = chatbot.data
labels = chatbot.label_encoder(data.tag)
patterns = data.patterns.values.tolist()
vocabulary = chatbot.get_vocabulary(patterns)
```

### Training

Das neuronale Netz wird mit den Bag-of-Words-Vektoren und den Labels trainiert.

```python
X_train = [chatbot.bag_of_words_as_vector(sentence, vocabulary) for sentence in patterns]
X_train = np.array(X_train)
y_train = np.eye(len(set(labels)))[labels]

input_size = len(vocabulary)
hidden_size = 64
output_size = len(set(labels))
epochs = 1000
learning_rate = 0.1

nn = SimpleNeuralNetwork(input_size, hidden_size, output_size, epochs, learning_rate)
input_weight, output_weight = nn.train(X_train, y_train)
```

### Vorhersage und Genauigkeit

Die trainierten Gewichte werden an den Entscheidungsbaum-Klassifikator übergeben. Anschließend werden Vorhersagen und die Genauigkeit berechnet.

```python
custom_classifier = CustomDecisionTreeClassifier()
custom_classifier.fit(input_weight, output_weight)

predictions = custom_classifier.predict(X_train)
accuracy = custom_classifier.accuracy_score(y_train, predictions)
print(f'Accuracy: {accuracy:.2f}%')
```

### Chatbot-Interaktion

Der Chatbot kann nun interaktiv genutzt werden. Die Eingabe wird vektorisiert und die passende Antwort ausgegeben.

```python
responses = list(zip(labels, responses))
while True:
    message = input('Message: ')
    if message.lower() == 'exit':
        break
    bow_message = [chatbot.bag_of_words(message, vocabulary)]
    prediction = custom_classifier.predict(bow_message)
    response_options = [tupel[1] for tupel in responses if tupel[0]==prediction]
    print(random.choice(response_options))
```

---

## Parameterwahl

- **input_size**: Länge des Vokabulars (Anzahl der Features)
- **hidden_size**: Anzahl der Neuronen in der versteckten Schicht (z.B. 8, 16, 32, 64)
- **output_size**: Anzahl der verschiedenen Tags (Intents)
- **epochs**: Anzahl der Trainingsdurchläufe (z.B. 300–1000)
- **learning_rate**: Lernrate (z.B. 0.01–0.1)

Die optimale Wahl hängt von der Größe und Komplexität deiner Daten ab.

---

## Beispiel

```python
input_size = len(vocabulary)
hidden_size = 32
output_size = len(set(labels))
epochs = 500
learning_rate = 0.05

nn = SimpleNeuralNetwork(input_size, hidden_size, output_size, epochs, learning_rate)
input_weight, output_weight = nn.train(X_train, y_train)

custom_classifier = CustomDecisionTreeClassifier()
custom_classifier.fit(input_weight, output_weight)

predictions = custom_classifier.predict(X_train)
accuracy = custom_classifier.accuracy_score(y_train, predictions)
print(f'Accuracy: {accuracy:.2f}%')
```

---

## Lizenz

Dieses Modul ist für Bildungszwecke gedacht.  
Bitte beachte die Lizenzbedingungen im Repository.

---

**Fragen oder Probleme?**  
Erstelle ein Issue im Repository oder kontaktiere den Autor.
