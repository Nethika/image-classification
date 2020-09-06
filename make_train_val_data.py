#divide all data to train and val only at the first time

import os
import shutil
import numpy as np


train_data = "data/train"
val_data = "data/val"
folders = os.listdir(train_data)

#create validation folders
for folder in folders:
    val_dir = os.path.join(val_data,folder)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

for folder in folders:
    train_path = os.path.join(train_data,folder)
    val_path = os.path.join(val_data,folder)

    files =  os.listdir(train_path)
    for f in files:
        if np.random.rand(1) < 0.25:
            shutil.move(train_path + '/'+ f, val_path + '/'+ f)