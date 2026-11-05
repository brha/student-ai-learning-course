# Week 3 Neural networks and simple deep learning

## Meeting 5

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
- Or read:
  - MNIST dataset: https://keras.io/api/datasets/mnist/
  - Sequential model: https://keras.io/api/models/sequential/
  - Dense layers: https://keras.io/api/layers/core_layers/dense/
  - Model training and evaluation: https://keras.io/api/models/model_training_apis/
- Activity: Use sample code and observe outputs
  - Use, understand and modify one of the following
    - **digit_classifier_starter_tensorflow.py**
    - **digit_classifier_starter_scikit-learn.py**

## Meeting 6

**Lesson 16:** Image classification introduction
- Goal: Explain image datasets and processing
  - Visual activity: 
    - TensorFlow’s cats-vs-dogs image-classification tutorial: https://www.tensorflow.org/tutorials/images/classification
    - It includes a Google Colab notebook: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb
  - Features:
    - Step-by-step guide for building an image classifier using Keras.
    - Walks through data loading, preprocessing, model creation/training, visualizing results, overfitting, tuning, and using techniques like data augmentation and dropout.
    - Clear code, plots, and explanations that are classroom and beginner friendly.
    - Dataset: Uses a filtered version of Cats vs. Dogs dataset (direct link in tutorial)
- Activity: Cat-vs-dog example
  - Use, understand and modify  **lesson16_image_data_Version2.py**

**Lesson 17:** Mini image classifier
- Goal: Train a basic model on small images
- Activity: Modify starter code
  - Use, understand and modify **lesson17_mini_image_classifier_Version2.py**

**Lesson 18:** Troubleshooting and model improvement
- Goal: Discuss overfitting and tuning
- Activity: Adjust epochs, layers, and data size
  - Use, understand and modify **lesson18_overfitting_and_tuning_Version2.py**

# Week 3 Starter Code

This folder contains neural network starter scripts for week 3.

- lesson13_neuron_analogy.py
- lesson14_neural_neetwork_starter.py
- digit_classifier_starter_scikit-learn.py
- digit_classifier_starter_tensorflow.py