# main.py

import os
from typing import List
import tensorflow as tf
import rasterio as io
from method import create_dataset
from Model import unet

# Constants for modes
PREDICT = True
TRAIN = False

# Configuration
PRE_TRAINED = True
MODE = PREDICT
BATCH_SIZE = 32
BUFFER_SIZE = 1000
TRAIN_SPLIT = 0.9
EPOCHS = 50

# Paths
TRAIN_PATH = "./exemple_data/input"
MASK_PATH = "./exemple_data/mask"
MODEL_SAVE_PATH = "./unet_ac_9857.hdf5"
PREDICTION_SAVE_PATH = "./predict"

def get_sorted_file_paths(directory: str) -> List[str]:
    """Retrieve sorted file paths from the specified directory."""
    paths = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            paths.append(path)
    return sorted(paths)

def split_dataset(paths: List[str], masks: List[str], split_ratio: float):
    """Splits the dataset into training and validation sets."""
    split_index = int(len(paths) * split_ratio)
    return (
        paths[:split_index], masks[:split_index],   # Training set
        paths[split_index:], masks[split_index:]    # Validation set
    )

def prepare_datasets(paths_train, masks_train, paths_valid, masks_valid):
    """Prepare TensorFlow datasets for training and validation."""
    train_dataset = create_dataset(paths_train, masks_train, train=True)
    valid_dataset = create_dataset(paths_valid, masks_valid)
    
    train_dataset = (
        train_dataset
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    
    return train_dataset, valid_dataset, len(train_dataset)

def load_model(pretrained: bool, save_path: str):
    """Load a U-Net model, optionally with pretrained weights."""
    if pretrained:
        return unet(save_path)
    return unet()

def save_predictions(results, paths_valid):
    """Save model predictions to disk as GeoTIFF files."""
    os.makedirs(PREDICTION_SAVE_PATH, exist_ok=True)
    
    for i, image_path in enumerate(paths_valid):
        tmp = io.open(image_path)
        prediction_path = os.path.join(PREDICTION_SAVE_PATH, f"{i}.tif")
        with io.open(prediction_path, 'w',
                     driver='GTiff',
                     height=results[i].shape[1],
                     width=results[i].shape[2],
                     count=1,
                     dtype=results[i].dtype,
                     transform=tmp.transform,
                     crs=tmp.crs) as dst:
            dst.write(results[i])

def main():
    # Print the current working directory and dataset sizes
    print("Current working directory:", os.path.abspath(os.getcwd()))
    print("Train set size:", len(os.listdir(TRAIN_PATH)))
    print("Train masks size:", len(os.listdir(MASK_PATH)))

    # Prepare paths
    image_paths = get_sorted_file_paths(TRAIN_PATH)
    mask_paths = get_sorted_file_paths(MASK_PATH)

    # Split the dataset into training and validation sets
    paths_train, masks_train, paths_valid, masks_valid = split_dataset(
        image_paths, mask_paths, TRAIN_SPLIT
    )

    # Prepare TensorFlow datasets
    train_dataset, valid_dataset, train_length = prepare_datasets(
        paths_train, masks_train, paths_valid, masks_valid
    )

    # Load the model
    model = load_model(PRE_TRAINED, MODEL_SAVE_PATH)

    if MODE == PREDICT:
        # Predict and save the results
        results = model.predict(valid_dataset)
        save_predictions(results, paths_valid)
    else:
        # Train the model
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor='loss', verbose=1, save_best_only=True
        )
        model.fit(
            train_dataset,
            steps_per_epoch=train_length // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[model_checkpoint]
        )

if __name__ == "__main__":
    main()
