# Lesson 16: Image classification — exploring and preparing images
#
# Read:
# TensorFlow image classification tutorial:
# https://www.tensorflow.org/tutorials/images/classification
# Loading images from directories:
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

# --------------------------------------------------
# Step 1: Import the tools
# --------------------------------------------------
import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt

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
train_dir = os.path.join(data_dir, "train")

print("Step 2: Dataset ready")
print("Training folders:", os.listdir(train_dir))

# --------------------------------------------------
# Step 3: Load the images from their folders
# --------------------------------------------------
# The folder names become the class labels.
# Students: Try changing image_size or batch_size.
image_size = (160, 160)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)

print("Step 3: Class names:", train_ds.class_names)

# --------------------------------------------------
# Step 4: Inspect one batch
# --------------------------------------------------
images, labels = next(iter(train_ds))

print("Step 4: Image batch shape:", images.shape)
print("Step 4: Label batch shape:", labels.shape)
print("Step 4: First few labels:", labels[:5].numpy().flatten())

# --------------------------------------------------
# Step 5: Display a few images and their labels
# --------------------------------------------------
# In this dataset, label 0 is cats and label 1 is dogs.
class_names = train_ds.class_names

plt.figure(figsize=(9, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    label_number = int(labels[i].numpy()[0])
    plt.title(class_names[label_number])
    plt.axis("off")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 6: Normalize the pixel values
# --------------------------------------------------
# Images start with pixel values from 0 to 255.
# Rescaling changes them to values from 0 to 1.
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

normalized_images = normalizer(images)

print("Step 6: Original pixel range:",
      float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))
print("Step 6: Normalized pixel range:",
      float(tf.reduce_min(normalized_images)),
      "to",
      float(tf.reduce_max(normalized_images)))

# --------------------------------------------------
# Step 7: Experiment
# --------------------------------------------------
# Students: Change image_size and batch_size in Step 3.
# How do the image shape and number of images per batch change?
# Why might images need to be the same size before model training?