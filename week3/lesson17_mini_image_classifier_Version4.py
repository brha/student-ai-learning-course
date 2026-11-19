# Lesson 17: Train a mini image classifier using local Cats and Dogs data
#
# Dataset download:
# https://www.microsoft.com/en-us/download/details.aspx?id=54765
#
# Read:
# TensorFlow image classification:
# https://www.tensorflow.org/tutorials/images/classification
# Keras Conv2D layer:
# https://keras.io/api/layers/convolution_layers/convolution2d/
# Keras model training:
# https://keras.io/api/models/model_training_apis/

# --------------------------------------------------
# Step 1: Import the tools
# --------------------------------------------------
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

tf.keras.utils.set_random_seed(42)

# --------------------------------------------------
# Step 2: Set the path and experiment settings
# --------------------------------------------------
# Change this path to the location of your extracted PetImages folder.
DATA_DIR = Path(
    r"C:\git\student-ai-learning-course\week3\kagglecatsanddogs_5340\PetImages"
)

# Students: Try changing these settings.
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.2
EPOCHS = 3

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Could not find the PetImages folder:\n{DATA_DIR}\n"
        "Check the DATA_DIR path."
    )

print("Step 2: Path for images found")

# --------------------------------------------------
# Step 3: Find and check the image files
# --------------------------------------------------
# The dataset can contain unreadable or corrupt images.
# This step skips files that Pillow cannot verify in the "Simple test"
# In "Better test" this step skips files that has a wrong interna file-type or Pillow cannot decode
# In "Best test" this step skips files that TensorFlow cannot decode
# It does not delete or modify any files.
# Try to remove "Better test" and "Best test" and see what happens

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

        if image_path.suffix.lower() not in {".jpg", ".jpeg", "webp", "png", "gif", "bmp"}:
            continue
        
        try:
            # Simple test
            with Image.open(image_path) as image:
                image.verify()      #Quick and simple test if image ok

            # Better test
            with Image.open(image_path) as image:
                if not (image.format in ("JPEG", "JPEG XL", "WebP", "PNG", "GIF", "BMP")):  #Be sure of the internal file type is ok
                    bad_files.append(str(image_path))
                    print(f"Skipping unreadable image: {image_path} ({image.format})")
                    continue
                image.load()   # Fully decode the image to catch damaged files.
                
            # Best test
            # Use TensorFlow's decoder during validation, since TensorFlow will
            # decode the images again when the model trains.
            image_bytes = tf.io.read_file(str(image_path))
            decoded = tf.io.decode_image(
                image_bytes,
                channels=3,
                expand_animations=False
            )
            # Force TensorFlow to actually decode the image now.
            # Otherwise, an error may not appear until model.fit().
            _ = decoded.numpy()

            valid_paths.append(str(image_path))
            labels.append(class_to_label[class_name])

        except (UnidentifiedImageError, OSError, ValueError, AssertionError) as error:
            bad_files.append(str(image_path))
            print(f"Skipping unreadable image: {image_path} ({error})")
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError) as error:
            bad_files.append(str(image_path))
            print(f"TensorFlow cannot decode: {image_path}")
            print(f"  Reason: {error}")

print("Step 3: Readable images found:", len(valid_paths))
print("Step 3: Unreadable images skipped:", len(bad_files))

if bad_files:
    print("Unreadable or non-JPEG files skipped:", len(bad_files))
    for path in bad_files[:20]:
        print(path)

if not valid_paths:
    raise RuntimeError(
        "No readable JPG images were found. Check the folder and its contents."
    )

# --------------------------------------------------
# Step 4: Split each class into training and validation data
# --------------------------------------------------
# Splitting each class separately helps keep cats and dogs in both sets.
# Students: Change VALIDATION_FRACTION to change the split.
random_generator = np.random.default_rng(seed=42)

train_paths = []
train_labels = []
validation_paths = []
validation_labels = []

for label in [0, 1]:
    class_indices = [
        i for i, current_label in enumerate(labels)
        if current_label == label
    ]
    random_generator.shuffle(class_indices)

    split_at = int(len(class_indices) * (1 - VALIDATION_FRACTION))
    train_indices = class_indices[:split_at]
    validation_indices = class_indices[split_at:]

    train_paths.extend(valid_paths[i] for i in train_indices)
    train_labels.extend(labels[i] for i in train_indices)

    validation_paths.extend(valid_paths[i] for i in validation_indices)
    validation_labels.extend(labels[i] for i in validation_indices)

print("Step 4: Training images:", len(train_paths))
print("Step 4: Validation images:", len(validation_paths))

# --------------------------------------------------
# Step 5: Load and prepare each image
# --------------------------------------------------
# Resize every image to the same dimensions.
# Labels are shaped as one-item lists to match the model's output shape.
def prepare_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)

    # Decode common image formats, including BMP, PNG and JPEG. decode_jpeg only decodes JPG-files
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.ensure_shape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

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
    .prefetch(AUTOTUNE)
)

validation_ds = tf.data.Dataset.from_tensor_slices(
    (validation_paths, validation_labels)
)
validation_ds = (
    validation_ds
    .map(prepare_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Step 6: Training and validation datasets are ready")

# --------------------------------------------------
# Step 7: Build the CNN model
# --------------------------------------------------
# The Rescaling layer changes pixel values from 0–255 to 0–1.
# Students: Try changing the number of filters (16, 32, or 64).
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    ),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),

    # Sigmoid gives a probability between 0 and 1.
    # Here, 0 represents Cat and 1 represents Dog.
    tf.keras.layers.Dense(1, activation="sigmoid")
])

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
print("Step 8: Model compiled")

# --------------------------------------------------
# Step 9: Train the model
# --------------------------------------------------
# Students: Try changing EPOCHS to 1, 5, or 8.
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS
)

print("Step 9: Training complete")

# --------------------------------------------------
# Step 10: Evaluate the model
# --------------------------------------------------
validation_loss, validation_accuracy = model.evaluate(validation_ds)

print("Step 10: Validation accuracy:", round(validation_accuracy, 3))
print("Step 10: Validation loss:", round(validation_loss, 3))

# --------------------------------------------------
# Step 11: Plot training and validation accuracy
# --------------------------------------------------
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.show()

# --------------------------------------------------
# Step 12: Experiment
# --------------------------------------------------
# Students: Change one setting at a time and compare the results:
# - EPOCHS
# - IMAGE_SIZE
# - BATCH_SIZE
# - The number of filters in the Conv2D layers
# - VALIDATION_FRACTION
#
# Discuss:
# - Does training accuracy differ from validation accuracy?
# - What happens when the model gets more epochs?
# - Does a larger model always give better results?