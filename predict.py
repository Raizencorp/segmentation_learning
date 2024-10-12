import tensorflow as tf
import os
from method import create_dataset_from_paths, rearrange_array_inverted
from Model import unet
import rasterio as io

predict_path = "/home/cyril.gainon/Recherche/Tensorflow/data/predict"
print(os.path.abspath(os.getcwd()))
print("Train set:  ", len(os.listdir(predict_path)))

# Collecting image paths
paths = []
for dirname, _, filenames in os.walk(predict_path):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        paths.append(path)
paths.sort()

# Selecting a small validation subset (0.2% of total)
paths_valid = []
i = 0
while i < (len(paths) * 0.002):
    paths_valid.append(paths[i])
    i += 1

# Create validation dataset
valid_dataset = create_dataset_from_paths(paths_valid, paths_valid).batch(32)

"""
Model Management
"""
save = '/home/cyril.gainon/Recherche/Tensorflow/data/history/dernier/unet_ac_9857.hdf5'

# Load the model using the unet function
model = unet(save)

# Predict on the validation dataset
results = model.predict(valid_dataset)

"""
Post-process results (Optional)
"""
# Optionally threshold the results if needed
# results[results > 0.5] = 1
# results[results < 0.5] = 0

# Save predictions as GeoTIFFs
i = 0
for image_path in paths_valid:
    tmp = io.open(image_path)
    w = rearrange_array_inverted(results[i])  # Adjusting array shape back for saving

    output_path = f'/home/cyril.gainon/Recherche/Tensorflow/data/prediction/{i}.tif'
    with io.open(
        output_path,
        'w',
        driver='GTiff',
        height=results[i].shape[0],
        width=results[i].shape[1],
        count=1,
        dtype=results[i].dtype,
        transform=tmp.transform,
        crs=tmp.crs
    ) as dst:
        dst.write(w)
    
    if i % 50 == 0:
        print(f"Progress: {i / len(paths_valid):.2%}")

    i += 1
