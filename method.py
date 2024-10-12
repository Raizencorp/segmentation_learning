import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rasterio
from typing import List, Tuple

def data_augmentation(carte_img: tf.Tensor, mask_img: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies random data augmentation to the input image and mask."""
    if tf.random.uniform(()) > 0.5:
        carte_img = tf.image.flip_left_right(carte_img)
        mask_img = tf.image.flip_left_right(mask_img)
    return carte_img, mask_img

def rearrange_array_rgb(array: np.ndarray) -> np.ndarray:
    """Rearranges a given RGB array to the shape (256, 256, 3)."""
    reshaped = np.reshape(array, (256, 256, 3))
    return np.transpose(reshaped, (1, 2, 0))

def rearrange_array_single_channel(array: np.ndarray) -> np.ndarray:
    """Rearranges a given single-channel array to the shape (256, 256, 1)."""
    reshaped = np.reshape(array, (256, 256, 1))
    return np.transpose(reshaped, (1, 2, 0))

def rearrange_array_rgb_swap(array: np.ndarray) -> np.ndarray:
    """Rearranges an RGB array by swapping axes to (256, 256, 3)."""
    return np.swapaxes(array, 0, -1)

def rearrange_array_single_channel_swap(array: np.ndarray) -> np.ndarray:
    """Rearranges a single-channel array by swapping axes to (256, 256, 1)."""
    return np.swapaxes(array, 0, -1)

def rearrange_array_inverted(array: np.ndarray) -> np.ndarray:
    """Inverts the axes of an array, useful for preprocessing back to original."""
    return np.swapaxes(array, -1, 0)

def preprocess_image(image_path: str) -> tf.Tensor:
    """Preprocesses an image for training by reading and normalizing it."""
    with rasterio.open(image_path) as image_file:
        image_data = rearrange_array_rgb(image_file.read() / 255.0)
    return tf.convert_to_tensor(image_data, dtype=tf.float32)

def preprocess_mask(mask_path: str) -> tf.Tensor:
    """Preprocesses a mask for training."""
    with rasterio.open(mask_path) as mask_file:
        mask_data = rearrange_array_single_channel(mask_file.read())
    return tf.convert_to_tensor(mask_data, dtype=tf.int8)

def preprocessing_images_and_masks(carte_paths: List[str], mask_paths: List[str]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """Preprocesses all images and masks into tensors."""
    images = []
    masks = []
    total_files = len(carte_paths)

    for i, (carte_path, mask_path) in enumerate(zip(carte_paths, mask_paths)):
        images.append(preprocess_image(carte_path))
        masks.append(preprocess_mask(mask_path))

        # Progress update every 10 files
        if i % 10 == 0 or i == total_files - 1:
            progress = (i + 1) / total_files * 100
            print(f"Loading images: {progress:.2f}%")

    return images, masks

def create_dataset_from_paths(image_paths: List[str], mask_paths: List[str], train: bool = False) -> tf.data.Dataset:
    """Creates a TensorFlow dataset from image and mask paths."""
    images, masks = preprocessing_images_and_masks(image_paths, mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))

    if train:
        dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def visualize(display_list: List[tf.Tensor]) -> None:
    """Displays input images, true masks, and predicted masks."""
    plt.figure(figsize=(15, 15))
    titles = ['Input Image', 'True Mask', 'Predicted Mask']

    for i, item in enumerate(display_list):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(item))
        plt.axis('off')
    
    plt.show()

def show_predictions(sample_image: tf.Tensor, sample_mask: tf.Tensor, model: tf.keras.Model) -> None:
    """Generates and visualizes predictions from the model for a given sample image and mask."""
    pred_mask = model.predict(sample_image[tf.newaxis, ...])
    visualize([sample_image, sample_mask, pred_mask])
