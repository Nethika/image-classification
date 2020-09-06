# USAGE
# python classify.py --model multi-label.model --image_path examples/sleeping.jpeg

# import the necessary packages
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils import paths
import os
import json
#import matplotlib
#import matplotlib.pyplot as plt


# Memory management
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


# Helper Functions
def classify(im_path, model, label_map):
    """ classifies images in a given folder using the 'model'"""
    print (im_path)
    print ("========================")
    input_img = load_img(im_path)
    img = load_img(im_path,target_size=(input_height, input_width))
    img = img_to_array(img)
    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)
    
    labels = list(label_map.keys())

    predictions = model.predict(img)[0]

    #plt.imshow(input_img)
    #plt.show()

    for cat in labels:
        category_index = label_map[cat]
        value = predictions[category_index]
        print("{}: {:.2f}%".format(cat, value * 100))
    print ("")

# Inputs
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to trained model model")
ap.add_argument("-i", "--image_path", required=True,
    help="path to input image directory")
args = vars(ap.parse_args())

# path to trained model 
model_path = args["model"]
# path to input image directory
im_path = args["image_path"]

# input image size
input_image_size = 128

input_width = input_image_size
input_height = input_image_size

# load the trained convolutional neural network
print("[INFO] loading model...")
model = load_model(model_path)
print ("[INFO] model loaded!")
print ("")

# Get Class Names
label_map = json.load(open("lebels.txt"))

"""
# if image directory
image_paths = list(paths.list_images(image_dir))

for img in image_paths:
    classify(img, model, label_map)
"""
# Classify
# if one image

classify(im_path, model, label_map)



