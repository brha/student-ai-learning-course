# Lesson 16: Image classification — exploring and preparing images
# This version loads Cats vs Dogs using TensorFlow Datasets (TFDS).
#
# Read:
# Cats vs Dogs dataset: https://www.tensorflow.org/datasets/catalog/cats_vs_dogs
# tfds.load: https://www.tensorflow.org/datasets/api_docs/python/tfds/load
# Image resizing: https://www.tensorflow.org/api_docs/python/tf/image/resize

# --------------------------------------------------
# Step 1: Import the tools
# --------------------------------------------------
import tensorflow as tf
# pip install tensorflow tensorflow-datasets matplotlib
# python -m pip install --upgrade importlib-resources
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# --------------------------------------------------
# Step 2: Choose image size and batch size
# --------------------------------------------------
# Students: Change these values and observe how the image shape changes.
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

# --------------------------------------------------
# Step 3: Load the dataset
# --------------------------------------------------
# TFDS downloads the dataset the first time this code runs.
# Here, 80% of the data is used for training and 20% for validation.
(train_raw, validation_raw), info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True,
    shuffle_files=True
)

print("Step 3: Dataset loaded")
print("Class names:", info.features["label"].names)
print("Training examples:", info.splits["train"].num_examples)

# --------------------------------------------------
# Step 4: Resize images and prepare labels
# --------------------------------------------------
# The images have different sizes. Resizing gives them a consistent shape.
# Labels are converted to integers: typically 0 = cat and 1 = dog.
def prepare_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    label = tf.cast(label, tf.int32)
    return image, label

train_prepared = train_raw.map(
    prepare_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_prepared = validation_raw.map(
    prepare_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

print("Step 4: Images resized to:", IMAGE_SIZE)

# --------------------------------------------------
# Step 5: Make batches
# --------------------------------------------------
# A batch is a group of images processed together.
# Prefetch helps prepare the next batch while the current one is being used.
train_ds = (
    train_prepared
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_ds = (
    validation_prepared
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("Step 5: Datasets prepared in batches")

# --------------------------------------------------
# Step 6: Inspect a batch
# --------------------------------------------------
images, labels = next(iter(train_ds))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First few labels:", labels[:5].numpy())

# --------------------------------------------------
# Step 7: Display a few images and their labels
# --------------------------------------------------
# Students: Change the number of images displayed or inspect another batch.
class_names = info.features["label"].names

plt.figure(figsize=(9, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)

    # Convert pixel values to integers for display.
    image_to_show = tf.cast(images[i], tf.uint8)
    plt.imshow(image_to_show)

    label_number = int(labels[i].numpy())
    plt.title(class_names[label_number])
    plt.axis("off")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 8: Normalize the pixel values
# --------------------------------------------------
# Pixel values are normally between 0 and 255.
# Rescaling converts them to values between 0 and 1.
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

normalized_images = normalizer(images)

print("Original pixel range:",
      float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))
print("Normalized pixel range:",
      float(tf.reduce_min(normalized_images)),
      "to",
      float(tf.reduce_max(normalized_images)))

# --------------------------------------------------
# Step 9: Experiment
# --------------------------------------------------
# Students: Change one setting at a time and run the code again.
# - Change IMAGE_SIZE.
# - Change BATCH_SIZE.
# - Display a different batch by calling next(iter(train_ds)) again.
# Discuss:
# - Why do images need consistent dimensions?
# - What is a label?
# - Why might pixel values be rescaled before training?