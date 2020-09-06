# USAGE
# python train_network.py --train_data data/train --val_data data/val --model multi-label.model

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from model_build import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob
from keras.models import load_model
import imutils
import argparse
import json

# Memory management
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# Helper Functions
def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

    
# Inputs

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-t", "--train_data", required=True,
    help="path to training dataset")
ap.add_argument("-v", "--val_data", required=True,
    help="path to validation dataset")
ap.add_argument("-s", "--image_size", type=int, default=128,
    help="input image size")
ap.add_argument("-m", "--model", type=str, default="multi-label.model",
    help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot-multi-label.png",
    help="path to output loss/accuracy plot")

args = vars(ap.parse_args())

# path to training dataset
train_data_dir = args["train_data"]

# path to validation dataset
validation_data_dir = args["val_data"]

# input_image_size
input_image_size = args["image_size"]

# path to output model
model_path = args["model"]

# path to output plot
plot_path = args["plot"]



# initialize the number of epochs to train for
EPOCHS = 25
# initialize learning rate
INIT_LR = 1e-3
# initialize batch size
BS = 32

# Input image size
input_width = input_image_size
input_height = input_image_size    
    
# Initiate the train and validation generators with data Augumentation 
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

val_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (input_height, input_width),
    batch_size = BS,
    class_mode = "categorical"
)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (input_height, input_width),
    batch_size = BS,
    class_mode = "categorical")

# Class names
label_map_train = (train_generator.class_indices)

# save class names dictionary
json.dump(label_map_train, open("lebels.txt",'w'))

# number of classes
num_classes = len(label_map_train)

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=input_width , height=input_height , depth=3, classes=num_classes)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] model compiled!")

# Number of Images
nb_train_samples = get_nb_files(train_data_dir)
nb_validation_samples = get_nb_files(validation_data_dir)

# Train the model 
print("[INFO] training the model...")
H = model.fit_generator(train_generator,
                        steps_per_epoch = nb_train_samples//BS,
                        epochs = EPOCHS,
                        verbose=1,
                        validation_data = validation_generator)
print("[INFO] model trained!")
# save the model to disk
print("[INFO] serializing network...")
model.save(model_path)
print("[INFO] model saved!")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training & Validation Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot_path)
#plt.show()
print("[INFO] plot saved!")
