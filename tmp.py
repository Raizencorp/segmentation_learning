tmp.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rasterio

def rearrange_arrayRGB(array):
    z = np.swapaxes(array, 0, -1)
    return z

def rearrange_array01(array):
    z = np.swapaxes(array,0,-1)
    return z
def rearrange_array02(array):
    z = np.swapaxes(array,-1,0)
    return z

def preprocessing(carte_path):
    i = 0
    l = []
    while i < len(carte_path):
        carte_img = rasterio.open(carte_path[i])
        tf_tmp = tf.convert_to_tensor(rearrange_arrayRGB(carte_img.read()/255))
        l.append(tf_tmp)

        pourcentage = i / len(carte_path)
        if i%500 == 0:
            print("Chargement des images :" + str(pourcentage*100) + "%")
        i += 1
    print("Chargement de images terminé")
    return l


def create_dataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(preprocessing(paths))
    return ds