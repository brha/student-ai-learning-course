# Lesson 17: Train a mini image classifier
#
# Read:
# TensorFlow image classification tutorial:
# https://www.tensorflow.org/tutorials/images/classification
# Keras Conv2D layer:
# https://keras.io/api/layers/convolution_layers/convolution2d/
# Keras model training:
# https://keras.io/api/models/model_training_apis/

# --------------------------------------------------
# Step 1: Import the tools
# --------------------------------------------------
import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)

# --------------------------------------------------
# Step 2: Download and extract the practice dataset
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
# Step 3: Load training and validation images
# --------------------------------------------------
# Training images teach the model.
# Validation images help us check it on examples it did not train on.
image_size = (160, 160)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, "validation"),
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary",
    shuffle=False
)

print("Step 3: Classes:", train_ds.class_names)

# --------------------------------------------------
# Step 4: Prepare the datasets for training
# --------------------------------------------------
# Prefetch helps prepare the next batch while the model is working.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)

# --------------------------------------------------
# Step 5: Build the CNN model
# --------------------------------------------------
# Students: Try changing the number of filters (16 and 32).
# The final sigmoid layer outputs a probability for the dog class.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(160, 160, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

print("Step 5: Model created")
model.summary()

# --------------------------------------------------
# Step 6: Choose how the model learns
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# Step 7: Train the model
# --------------------------------------------------
# Students: Try changing epochs to 1, 3, or 8.
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=3
)

# --------------------------------------------------
# Step 8: Evaluate on validation images
# --------------------------------------------------
validation_loss, validation_accuracy = model.evaluate(validation_ds)

print("Step 8: Validation accuracy:", round(validation_accuracy, 3))
print("Step 8: Validation loss:", round(validation_loss, 3))

# --------------------------------------------------
# Step 9: Plot training and validation accuracy
# --------------------------------------------------
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and validation accuracy")
plt.show()

# --------------------------------------------------
# Step 10: Experiment
# --------------------------------------------------
# Students: Change one setting at a time:
# - epochs
# - number of filters in the Conv2D layers
# - image_size in Step 3 (also update Input shape in Step 5)
# What happens to training time and validation accuracy?