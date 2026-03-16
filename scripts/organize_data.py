"""
Organize Kaggle Dogs vs Cats train data into cats/ and dogs/ subdirectories.
Run from project root: python scripts/organize_data.py
Expects data/raw/train/train/*.jpg with filenames like cat.0.jpg, dog.1.jpg.
"""
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RAW_TRAIN_DIR = Path("data/raw/train/train")
OUTPUT_CATS_DIR = Path("data/raw/train/cats")
OUTPUT_DOGS_DIR = Path("data/raw/train/dogs")

def main():
    OUTPUT_CATS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DOGS_DIR.mkdir(parents=True, exist_ok=True)
    cat_count = 0
    dog_count = 0
    for img_file in RAW_TRAIN_DIR.glob("*.jpg"):
        if img_file.name.startswith("cat."):
            shutil.move(str(img_file), str(OUTPUT_CATS_DIR / img_file.name))
            cat_count += 1
        elif img_file.name.startswith("dog."):
            shutil.move(str(img_file), str(OUTPUT_DOGS_DIR / img_file.name))
            dog_count += 1
        if (cat_count + dog_count) % 1000 == 0:
            logger.info("Processed: cats=%d, dogs=%d", cat_count, dog_count)
    logger.info("Done: cats=%d, dogs=%d", cat_count, dog_count)


if __name__ == "__main__":
    main()
