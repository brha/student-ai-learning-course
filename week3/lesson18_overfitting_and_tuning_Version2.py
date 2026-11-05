# Lesson 18: Troubleshooting, overfitting, and model improvement
#
# Read:
# TensorFlow guide to overfitting and underfitting:
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
# Keras Dropout layer:
# https://keras.io/api/layers/regularization_layers/dropout/
# TensorFlow image classification tutorial:
# https://www.tensorflow.org/tutorials/images/classification

# --------------------------------------------------
# Step 1: Import the tools
# --------------------------------------------------
import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)

# --------------------------------------------------
# Step 2: Choose experiment settings
# --------------------------------------------------
# Students: Change one setting at a time and compare the results.
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 8
TRAIN_BATCHES_TO_USE = 10  # Set to None to use all training data.
DROPOUT_RATE = 0.3

# --------------------------------------------------
# Step 3: Download and extract the practice dataset
# --------------------------------------------------
url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = tf.keras.utils.get_file(
    "cats_and_dogs_filtered.zip",
    origin=url
)

extract_path = os.path.join(os.path.dirname(zip_path), "cats_dogs_classroom")

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

data_dir = os.path.join(extract_path, "cats_and_dogs_filtered")

# --------------------------------------------------
# Step 4: Load training and validation images
# --------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, "validation"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# --------------------------------------------------
# Step 5: Optionally use only some of the training data
# --------------------------------------------------
# Using fewer batches makes training faster and lets students
# explore how a smaller dataset can affect model performance.
if TRAIN_BATCHES_TO_USE is not None:
    train_ds = train_ds.take(TRAIN_BATCHES_TO_USE)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)

print("Step 5: Training data batches:", tf.data.experimental.cardinality(train_ds).numpy())

# --------------------------------------------------
# Step 6: Build a model with dropout
# --------------------------------------------------
# Dropout turns off some neurons during training, which can help
# reduce overfitting. Try setting DROPOUT_RATE to 0.0 or 0.5.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

print("Step 6: Model created")
model.summary()

# --------------------------------------------------
# Step 7: Compile and train the model
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS
)

# --------------------------------------------------
# Step 8: Evaluate the model
# --------------------------------------------------
validation_loss, validation_accuracy = model.evaluate(validation_ds)

print("Step 8: Validation accuracy:", round(validation_accuracy, 3))
print("Step 8: Validation loss:", round(validation_loss, 3))

# --------------------------------------------------
# Step 9: Compare training and validation curves
# --------------------------------------------------
# If training accuracy keeps rising while validation accuracy stalls
# or falls, the model may be overfitting.
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 10: Investigate and improve
# --------------------------------------------------
# Try experiments and record the results:
# 1. Set TRAIN_BATCHES_TO_USE to 2, then to 10, then to None.
# 2. Change EPOCHS, for example to 2, 8, or 15.
# 3. Change DROPOUT_RATE to 0.0, 0.3, or 0.5.
# 4. Compare training accuracy with validation accuracy.
# Which change helped validation performance? Which made overfitting worse?