# Lesson 15: Digit classifier starter
#
# scikit-learn Digits dataset (8x8 images):
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
#
# MLPClassifier:
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#
# Confusion matrix:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

# --------------------------------------------------
# Step 1: Import the tools
# --------------------------------------------------
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --------------------------------------------------
# Step 2: Load the digit images and labels
# --------------------------------------------------
digits = load_digits()
X = digits.data
y = digits.target

print("Step 2: Number of images:", len(X))
print("Step 2: Image data shape:", X.shape)
print("Step 2: Labels include digits:", digits.target_names)

# --------------------------------------------------
# Step 3: Display one example image
# --------------------------------------------------
# Students: Change image_number to display another image.
image_number = 0

plt.imshow(digits.images[image_number], cmap="gray")
plt.title(f"Label: {y[image_number]}")
plt.axis("off")
plt.show()

# --------------------------------------------------
# Step 4: Split into training and testing data
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Step 4: Training images:", len(X_train))
print("Step 4: Testing images:", len(X_test))

# --------------------------------------------------
# Step 5: Create the classifier
# --------------------------------------------------
# The pipeline scales the pixel values and then trains a neural network.
# Students: Try changing hidden_layer_sizes, for example to (32,) or (64, 32).
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=500,
        random_state=42
    )
)

# --------------------------------------------------
# Step 6: Train the classifier
# --------------------------------------------------
model.fit(X_train, y_train)
print("Step 6: Model trained")

# --------------------------------------------------
# Step 7: Predict the digits in the test set
# --------------------------------------------------
predictions = model.predict(X_test)

print("Step 7: First 10 actual labels:    ", y_test[:10])
print("Step 7: First 10 predicted labels: ", predictions[:10])

# --------------------------------------------------
# Step 8: Evaluate the classifier
# --------------------------------------------------
accuracy = accuracy_score(y_test, predictions)
matrix = confusion_matrix(y_test, predictions)

print("Step 8: Accuracy:", round(accuracy, 3))
print("Step 8: Confusion matrix:")
print(matrix)

# --------------------------------------------------
# Step 9: Try a prediction and display its image
# --------------------------------------------------
# Students: Change test_number to inspect another test example.
test_number = 0

predicted_digit = model.predict([X_test[test_number]])[0]

plt.imshow(X_test[test_number].reshape(8, 8), cmap="gray")
plt.title(
    f"Actual: {y_test[test_number]}, "
    f"Predicted: {predicted_digit}"
)
plt.axis("off")
plt.show()

# --------------------------------------------------
# Step 10: Experiment
# --------------------------------------------------
# Students: Change one setting at a time and run the code again.
# - hidden_layer_sizes
# - test_size
# - random_state
# - test_number
# Compare the accuracy and confusion matrix.