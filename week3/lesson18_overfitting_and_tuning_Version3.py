# Lesson 18: Troubleshooting, overfitting, and model improvement
# Uses local Cats and Dogs images from the Microsoft dataset.
#
# Dataset:
# https://www.microsoft.com/en-us/download/details.aspx?id=54765
#
# Read:
# Overfitting and underfitting:
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
# TensorFlow image decoding:
# https://www.tensorflow.org/api_docs/python/tf/io/decode_image
# Keras Dropout:
# https://keras.io/api/layers/regularization_layers/dropout/

# --------------------------------------------------
# Step 1: Import the tools
# --------------------------------------------------
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)

# --------------------------------------------------
# Step 2: Choose the data path and experiment settings
# --------------------------------------------------
# Change this to the location of your extracted PetImages folder.
DATA_DIR = Path(
    r"C:\git\student-ai-learning-course\week3\kagglecatsanddogs_5340\PetImages"
)

# Students: Change one setting at a time and compare the results.
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.2
EPOCHS = 8

# Set to a number to limit training data and speed up experiments.
# Set to None to use all available training images.
TRAIN_BATCHES_TO_USE = 10

# Try 0.0, 0.3, or 0.5 to see how dropout affects the model.
DROPOUT_RATE = 0.3

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Could not find the PetImages folder:\n{DATA_DIR}\n"
        "Check the DATA_DIR path."
    )

# --------------------------------------------------
# Step 3: Find images and validate them with TensorFlow
# --------------------------------------------------
# A file extension does not guarantee that the contents are a valid image.
# This checks each file using TensorFlow's decoder before model.fit().
# It catches unreadable files now, rather than during training.
class_names = ["Cat", "Dog"]
class_to_label = {"Cat": 0, "Dog": 1}

valid_paths = []
labels = []
bad_files = []

for class_name in class_names:
    class_dir = DATA_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Could not find class folder:\n{class_dir}")

    for image_path in class_dir.iterdir():
        if not image_path.is_file():
            continue

        try:
            image_bytes = tf.io.read_file(str(image_path))
            decoded_image = tf.io.decode_image(
                image_bytes,
                channels=3,
                expand_animations=False
            )

            # Force TensorFlow to decode the image during this scan.
            # Otherwise an error might not appear until model.fit().
            _ = decoded_image.numpy()

            valid_paths.append(str(image_path))
            labels.append(class_to_label[class_name])

        except tf.errors.InvalidArgumentError as error:
            bad_files.append(str(image_path))
            print(f"Invalid or unsupported image: {image_path}")
            print(f"  Details: {error.message}")

        except tf.errors.NotFoundError as error:
            bad_files.append(str(image_path))
            print(f"File not found or inaccessible: {image_path}")
            print(f"  Details: {error.message}")

        except tf.errors.OpError as error:
            # Handles other TensorFlow file or decoding errors.
            bad_files.append(str(image_path))
            print(f"TensorFlow could not read: {image_path}")
            print(f"  Details: {error.message}")

print()
print("Step 3: Valid images:", len(valid_paths))
print("Step 3: Images skipped:", len(bad_files))

if not valid_paths:
    raise RuntimeError(
        "No readable images were found. Check the folder path and contents."
    )

# Count the number of examples in each class.
print("Cats:", labels.count(class_to_label["Cat"]))
print("Dogs:", labels.count(class_to_label["Dog"]))

# --------------------------------------------------
# Step 4: Split each class into training and validation data
# --------------------------------------------------
# Splitting the classes separately helps ensure both sets contain cats and dogs.
# Students: Change VALIDATION_FRACTION and observe the effect.
rng = np.random.default_rng(seed=42)

train_paths = []
train_labels = []
validation_paths = []
validation_labels = []

for label in [class_to_label["Cat"], class_to_label["Dog"]]:
    class_indices = [
        index for index, current_label in enumerate(labels)
        if current_label == label
    ]

    rng.shuffle(class_indices)

    split_at = int(len(class_indices) * (1 - VALIDATION_FRACTION))
    train_indices = class_indices[:split_at]
    validation_indices = class_indices[split_at:]

    train_paths.extend(valid_paths[index] for index in train_indices)
    train_labels.extend(labels[index] for index in train_indices)

    validation_paths.extend(valid_paths[index] for index in validation_indices)
    validation_labels.extend(labels[index] for index in validation_indices)

print()
print("Step 4: Training images:", len(train_paths))
print("Step 4: Validation images:", len(validation_paths))

if not train_paths or not validation_paths:
    raise RuntimeError(
        "The training or validation set is empty. "
        "Check the dataset and VALIDATION_FRACTION."
    )

# --------------------------------------------------
# Step 5: Decode and prepare images for the model
# --------------------------------------------------
# decode_image supports common image formats, including JPEG, PNG, GIF, and BMP.
# Images are resized to the same dimensions so they can be placed in batches.
def prepare_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    # Tell TensorFlow that decoded images have 3 color channels.
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.ensure_shape(
        image,
        [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
    )

    # The model's final layer returns one probability per image.
    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, (1,))

    return image, label

# --------------------------------------------------
# Step 6: Create training and validation datasets
# --------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)
train_ds = (
    train_ds
    .shuffle(buffer_size=min(len(train_paths), 1000))
    .map(prepare_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
)

# Optionally limit the number of training batches for quicker experiments.
if TRAIN_BATCHES_TO_USE is not None:
    train_ds = train_ds.take(TRAIN_BATCHES_TO_USE)

train_ds = train_ds.prefetch(AUTOTUNE)

validation_ds = tf.data.Dataset.from_tensor_slices(
    (validation_paths, validation_labels)
)
validation_ds = (
    validation_ds
    .map(prepare_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print()
print("Step 6: Datasets are ready")
print(
    "Training batches used:",
    tf.data.experimental.cardinality(train_ds).numpy()
)

# --------------------------------------------------
# Step 7: Build the CNN model
# --------------------------------------------------
# Students: Try changing the number of filters or add/remove a layer.
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    ),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),

    # Dropout randomly turns off some neurons during training.
    # Students: Compare DROPOUT_RATE values such as 0.0 and 0.5.
    tf.keras.layers.Dropout(DROPOUT_RATE),

    # Output is a probability: near 0 = Cat, near 1 = Dog.
    tf.keras.layers.Dense(1, activation="sigmoid")
])

print()
print("Step 7: Model created")
model.summary()

# --------------------------------------------------
# Step 8: Configure how the model learns
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# Step 9: Train the model
# --------------------------------------------------
# Students: Try changing EPOCHS.
# Compare the training and validation results after each experiment.
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS
)

print("Step 9: Training complete")

# --------------------------------------------------
# Step 10: Evaluate the model on validation images
# --------------------------------------------------
validation_loss, validation_accuracy = model.evaluate(validation_ds)

print()
print("Step 10: Validation accuracy:", round(validation_accuracy, 3))
print("Step 10: Validation loss:", round(validation_loss, 3))

# --------------------------------------------------
# Step 11: Plot training and validation results
# --------------------------------------------------
# If training accuracy keeps increasing while validation accuracy stalls
# or decreases, that can be a sign of overfitting.
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
# Step 12: Experiment and reflect
# --------------------------------------------------
# Students: Change one setting at a time:
# - TRAIN_BATCHES_TO_USE: try 2, 10, or None
# - EPOCHS: try 2, 8, or 15
# - DROPOUT_RATE: try 0.0, 0.3, or 0.5
# - BATCH_SIZE: try 16 or 64
# - Number of filters in the Conv2D layers
#
# Questions:
# - Does training accuracy differ from validation accuracy?
# - What happens when the model gets more epochs?
# - Does using more training data change the validation results?
# - Does a larger model always perform better?