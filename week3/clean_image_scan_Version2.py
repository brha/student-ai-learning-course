# Use this to save the good files to new directories 
# and remove the check in the programs for lesson 16, 17 and 18

from pathlib import Path
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

# DATA_DIR should point to the original PetImages folder.
# Change this path to the location of your extracted PetImages folder.
DATA_DIR = Path(
    r"C:\git\student-ai-learning-course\week3\kagglecatsanddogs_5340\PetImages"
)
# CLEAN_DIR will hold the verified, re-encoded copies.
CLEAN_DIR = DATA_DIR.parent / "PetImages_clean"

class_names = ["Cat", "Dog"]
class_to_label = {"Cat": 0, "Dog": 1}

valid_paths = []
labels = []
bad_files = []

for class_name in class_names:
    source_dir = DATA_DIR / class_name
    clean_class_dir = CLEAN_DIR / class_name
    clean_class_dir.mkdir(parents=True, exist_ok=True)

    for image_path in source_dir.iterdir():
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in {".jpg", ".jpeg", "webp", "png", "gif", "bmp"}:
            continue

        try:
            # Tjek the internal image-type
            with Image.open(image_path) as image:
                if not (image.format in ("JPEG", "JPEG XL", "WebP", "PNG", "GIF", "BMP")):  #Be sure of the internal file type is ok
                    continue

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

            # Fully decode, convert to standard RGB, and save a fresh JPEG.
            with Image.open(image_path) as image:
                image.load()
                clean_image = image.convert("RGB")

            clean_path = clean_class_dir / f"{image_path.stem}.jpg"
            clean_image.save(clean_path, format="JPEG", quality=95)

            valid_paths.append(str(clean_path))
            labels.append(class_to_label[class_name])

        except (UnidentifiedImageError, OSError, ValueError, AssertionError) as error:
            bad_files.append(str(image_path))
            print(f"Skipping unreadable image: {image_path} — {error}")
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError) as error:
            bad_files.append(str(image_path))
            print(f"TensorFlow cannot decode: {image_path} - {error}")

print("Readable images re-encoded:", len(valid_paths))
print("Unreadable images skipped:", len(bad_files))