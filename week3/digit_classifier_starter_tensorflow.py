# Lesson 15: Digit classifier with TensorFlow/Keras
#
# This program trains a neural network to recognize handwritten digits (0–9).
# The first run downloads the MNIST dataset.
#
# Read:
# MNIST dataset: https://keras.io/api/datasets/mnist/
# Sequential model: https://keras.io/api/models/sequential/
# Dense layers: https://keras.io/api/layers/core_layers/dense/
# Model training and evaluation: https://keras.io/api/models/model_training_apis/

# --------------------------------------------------
# Step 1: Import the tools
# --------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt

# Use a fixed seed to make results more repeatable.
tf.keras.utils.set_random_seed(42)

# --------------------------------------------------
# Step 2: Load the MNIST dataset
# --------------------------------------------------
# MNIST includes 60,000 training images and 10,000 test images.
# Each image is 28 by 28 pixels. Each label is a digit from 0 to 9.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Step 2: Dataset loaded")
print("Training images:", x_train.shape)
print("Test images:", x_test.shape)

# --------------------------------------------------
# Step 3: Look at an image and its label
# --------------------------------------------------
# Students: Change image_number to look at another training image.
image_number = 0

plt.imshow(x_train[image_number], cmap="gray")
plt.title(f"Label: {y_train[image_number]}")
plt.axis("off")
plt.show()

# --------------------------------------------------
# Step 4: Prepare the pixel values
# --------------------------------------------------
# Original pixel values range from 0 to 255.
# Dividing by 255 changes them to a range from 0 to 1.
# This often helps neural networks learn more effectively.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Step 4: Pixel values prepared")
print("Pixel value range:", x_train.min(), "to", x_train.max())

# --------------------------------------------------
# Step 5: Build the neural network
# --------------------------------------------------
# Flatten changes each 28x28 image into a list of 784 values.
# Dense layers learn patterns that help identify the digits.
#
# Students: Try changing 128 to 32 or 256.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

print("Step 5: Model created")
model.summary()

# --------------------------------------------------
# Step 6: Choose how the model learns and is measured
# --------------------------------------------------
# The labels are whole numbers (0 to 9), so we use
# sparse_categorical_crossentropy as the loss function.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# Step 7: Train the model
# --------------------------------------------------
# An epoch is one pass through the training data.
# Students: Try changing epochs to 1, 3, or 8.
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)

print("Step 7: Model trained")

# --------------------------------------------------
# Step 8: Evaluate the model on test images
# --------------------------------------------------
# The test images were not used to train the model.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("Step 8: Test results")
print("Test loss:", round(test_loss, 3))
print("Test accuracy:", round(test_accuracy, 3))

# --------------------------------------------------
# Step 9: Make predictions for the test images
# --------------------------------------------------
# The model returns a probability for each digit.
# argmax selects the digit with the highest probability.
probabilities = model.predict(x_test, verbose=0)
predicted_digits = tf.argmax(probabilities, axis=1)

print("Step 9: First 10 actual labels:   ", y_test[:10])
print("Step 9: First 10 predicted labels:", predicted_digits[:10].numpy())

# --------------------------------------------------
# Step 10: Display a confusion matrix
# --------------------------------------------------
# Rows represent the actual digits.
# Columns represent the digits predicted by the model.
confusion_matrix = tf.math.confusion_matrix(
    y_test,
    predicted_digits,
    num_classes=10
)

print("Step 10: Confusion matrix")
print(confusion_matrix.numpy())

# --------------------------------------------------
# Step 11: Inspect one test prediction
# --------------------------------------------------
# Students: Change test_number to inspect other test images.
test_number = 0
predicted_digit = int(predicted_digits[test_number])

plt.imshow(x_test[test_number], cmap="gray")
plt.title(
    f"Actual: {y_test[test_number]}, "
    f"Predicted: {predicted_digit}"
)
plt.axis("off")
plt.show()

# --------------------------------------------------
# Step 12: Experiment
# --------------------------------------------------
# Change one thing at a time, then run the program again:
# - Change epochs in Step 7.
# - Change the number of neurons (128) in Step 5.
# - Add another Dense layer, for example:
#       tf.keras.layers.Dense(64, activation="relu"),
# - Change image_number or test_number to inspect other images.
# Compare the test accuracy and confusion matrix after each change.