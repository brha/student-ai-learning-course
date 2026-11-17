# Lesson 16: Image classification — exploring and preparing images
# Uses the Cats and Dogs dataset downloaded from Microsoft.
#
# Microsoft dataset download:
# https://www.microsoft.com/en-us/download/details.aspx?id=54765
#
# Read:
# Loading images with TensorFlow:
# https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg
# Resizing images:
# https://www.tensorflow.org/api_docs/python/tf/image/resize

# --------------------------------------------------
# Step 1: Import the tools
# --------------------------------------------------
TF_ENABLE_ONEDNN_OPTS=0

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------
# Step 2: Set the path to the PetImages folder
# --------------------------------------------------
# Change this if you extracted the dataset somewhere else.
# Use a raw string (r"...") for Windows paths.
DATA_DIR = Path(r"C:\git\student-ai-learning-course\week3\kagglecatsanddogs_5340\PetImages")

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Could not find this folder:\n{DATA_DIR}\n"
        "Check DATA_DIR and make sure the archive is extracted."
    )

class_names = ["Cat", "Dog"]
class_to_label = {"Cat": 0, "Dog": 1}

print("Step 2: Dataset folder found:", DATA_DIR)

# --------------------------------------------------
# Step 3: Find image files and check that they can be opened
# --------------------------------------------------
# The dataset may contain a few corrupt or non-image files.
# This step checks the images and skips files that Pillow cannot verify.
# It does not delete or change any files.
valid_paths = []
labels = []
bad_files = []

for class_name in class_names:
    class_dir = DATA_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Could not find class folder: {class_dir}")

    for image_path in class_dir.iterdir():
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

            valid_paths.append(str(image_path))
            labels.append(class_to_label[class_name])

        except (UnidentifiedImageError, OSError, ValueError):
            bad_files.append(str(image_path))

print("Step 3: Valid images found:", len(valid_paths))
print("Step 3: Files skipped because they could not be verified:", len(bad_files))

if not valid_paths:
    raise RuntimeError(
        "No readable JPG images were found. Check the folder path and contents."
    )

# Show a few skipped filenames so students can discuss data quality.
if bad_files:
    print("Examples of skipped files:", bad_files[:5])

# --------------------------------------------------
# Step 4: Shuffle and split the image paths into two groups
# --------------------------------------------------
# This is only an example split for exploring the data.
# Students: Change the fraction to change how many images go into each group.
VALIDATION_FRACTION = 0.2
random_generator = np.random.default_rng(seed=42)
indices = random_generator.permutation(len(valid_paths))

split_at = int(len(indices) * (1 - VALIDATION_FRACTION))
train_indices = indices[:split_at]
validation_indices = indices[split_at:]

train_paths = [valid_paths[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]

validation_paths = [valid_paths[i] for i in validation_indices]
validation_labels = [labels[i] for i in validation_indices]

print("Step 4: Training image paths:", len(train_paths))
print("Step 4: Validation image paths:", len(validation_paths))

# --------------------------------------------------
# Step 5: Choose image size and batch size
# --------------------------------------------------
# All images need the same dimensions to be grouped into batches.
# Students: Try changing IMAGE_SIZE or BATCH_SIZE.
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

# --------------------------------------------------
# Step 6: Read, decode, and resize each image
# --------------------------------------------------
def prepare_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)

    # Decode JPEG images and make sure every image has 3 color channels.
    image = tf.io.decode_jpeg(image_bytes, channels=3)

    # Resize the image to the size selected in Step 5.
    image = tf.image.resize(image, IMAGE_SIZE)

    # Keep pixel values in the 0–255 range for now.
    label = tf.cast(label, tf.int32)
    return image, label

# --------------------------------------------------
# Step 7: Create TensorFlow datasets
# --------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
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

print("Step 7: TensorFlow datasets are ready")

# --------------------------------------------------
# Step 8: Inspect one batch
# --------------------------------------------------
images, batch_labels = next(iter(train_ds))

print("Step 8: Image batch shape:", images.shape)
print("Step 8: Label batch shape:", batch_labels.shape)
print("Step 8: First few labels:", batch_labels[:5].numpy())

# --------------------------------------------------
# Step 9: Display a few images and their labels
# --------------------------------------------------
plt.figure(figsize=(9, 6))

for i in range(min(6, len(images))):
    plt.subplot(2, 3, i + 1)
    plt.imshow(tf.cast(images[i], tf.uint8).numpy())
    plt.title(class_names[int(batch_labels[i].numpy())])
    plt.axis("off")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 10: Normalize the pixel values
# --------------------------------------------------
# Pixel values range from 0 to 255.
# Dividing by 255 changes them to a range from 0 to 1.
normalizer = tf.keras.layers.Rescaling(1.0 / 255)
normalized_images = normalizer(images)

print("Step 10: Original pixel range:",
      float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))
print("Step 10: Normalized pixel range:",
      float(tf.reduce_min(normalized_images)),
      "to",
      float(tf.reduce_max(normalized_images)))

# --------------------------------------------------
# Step 11: Experiment
# --------------------------------------------------
# Students: Change one setting at a time and run the code again.
# - Change IMAGE_SIZE.
# - Change BATCH_SIZE.
# - Change VALIDATION_FRACTION.
# - Inspect the skipped filenames.
#
# Discuss:
# - Why do images need consistent dimensions?
# - What do the labels 0 and 1 represent?
# - Why might some files need to be skipped?
# - Why are pixel values often normalized before training?