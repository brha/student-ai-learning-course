# Week 3 Neural networks and simple deep learning

#### Meeting 5

**Lesson 13:** Neural networks explained simply
- Goal: Explain neurons and layers
- Visual activity: 
  - https://playground.tensorflow.org
- Read: 
  - https://developers.google.com/machine-learning/crash-course/neural-networks
- Activity: Visual diagram and hands-on analogy
  - Use, understand and modify **lesson13_neuron_analogy.py**

**Lesson 14**: Handwritten digit recognition
- Goal: Show a classic beginner project
- Read:
  - make_moons dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
  - MLPClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
  - train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Activity: Introduce MNIST dataset
  - Use, understand and modify **lesson13_neural_network_starter.py**

**Lesson 15**: Build a simple classifier with TensorFlow/Keras or scikit-learn
- Goal: Train first neural network
- Read:
  - scikit-learn Digits dataset (8x8 images): https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
  - MLPClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
  - Confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- Activity: Use sample code and observe outputs
  - Use, understand and modify one of the following
    - **digit_classifier_starter_tensorflow.py**
    - **digit_classifier_starter_scikit-learn.py**

# Week 3 Starter Code

This folder contains a simple neural network starter script.

## Digit Recognition with Keras

digit_classifier_starter.py

## Challenge

- Change the number of neurons
- Try more epochs
- Visualize one sample image
- Add a prediction for a single test image
