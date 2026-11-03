# Lesson 14: Train a small neural network
#
# Read:
# make_moons dataset:
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
#
# MLPClassifier:
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#
# train_test_split:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# --------------------------------------------------
# Step 1: Import the tools
# --------------------------------------------------
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Step 2: Make a practice dataset
# --------------------------------------------------
# Each example has two input values and belongs to class 0 or 1.
# Students: Change noise to make the groups more or less mixed.
X, y = make_moons(n_samples=400, noise=0.20, random_state=4)

print("Step 2: Number of examples:", len(X))
print("Step 2: First input and label:", X[0], y[0])

# --------------------------------------------------
# Step 3: Split data into training and testing sets
# --------------------------------------------------
# The model learns from the training set.
# The test set is held back to check how well it works on new examples.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=4,
    stratify=y
)

print("Step 3: Training examples:", len(X_train))
print("Step 3: Testing examples:", len(X_test))

# --------------------------------------------------
# Step 4: Create the model
# --------------------------------------------------
# This pipeline scales the inputs before giving them to the neural network.
# hidden_layer_sizes=(8,) means one hidden layer with 8 neurons.
#
# Students: Try (4,), (16,), or (8, 8).
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(8,),
        max_iter=1500,
        random_state=4
    )
)

# --------------------------------------------------
# Step 5: Train the model
# --------------------------------------------------
model.fit(X_train, y_train)
print("Step 5: Model trained")

# --------------------------------------------------
# Step 6: Make predictions
# --------------------------------------------------
predictions = model.predict(X_test)
print("Step 6: First 10 predictions:", predictions[:10])

# --------------------------------------------------
# Step 7: Check the accuracy
# --------------------------------------------------
accuracy = accuracy_score(y_test, predictions)
print("Step 7: Test accuracy:", round(accuracy, 3))

# --------------------------------------------------
# Step 8: Experiment
# --------------------------------------------------
# Students: Change one setting at a time and run the code again.
# - noise in make_moons
# - hidden_layer_sizes
# - test_size
# What changed in the accuracy?