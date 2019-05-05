import os
import sys
import zipfile
import random
import tensorflow as tf
import time
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from shutil import copyfile
from resnet import model2


print(len(os.listdir('./PetImages/Cat/')))
print(len(os.listdir('./PetImages/Dog/')))

try:
    os.makedirs("training/dogs")
    os.makedirs("training/cats")
    os.makedirs("validation/dogs")
    os.makedirs("validation/cats")
except OSError as e:
    print ("Exception: ",e)
    pass


# Write a python function called split_data which takes
# a SOURCE directory containing the files
# a TRAINING directory that a portion of the files will be copied to
# a TESTING directory that a portion of the files will be copie to
# a SPLIT SIZE to determine the portion
# The files should also be randomized, so that the training set is a random
# X% of the files, and the test set is the remaining files
# SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
# Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
# and 10% of the images will be copied to the TESTING dir
# Also -- All images should be checked, and if they have a zero file length,
# they will not be copied over
#
# os.listdir(DIRECTORY) gives you a listing of the contents of that directory
# os.path.getsize(PATH) gives you the size of the file
# copyfile(source, destination) copies a file from source to destination
# random.sample(list, len(list)) shuffles a list
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = os.listdir(SOURCE)#[:1000]
    all_size = len(files)
    random.sample(files, all_size)
    split_point = int(all_size*SPLIT_SIZE)
    for f in files[:split_point]:
       src = os.path.join(SOURCE, f)
       dst = os.path.join(TRAINING, f)
       if os.path.getsize(src) > 0:
          copyfile(src, dst)  
#          print("copying: ", f)
       else:
          print("skipping empty file: ", f)

    for f in files[split_point:]:
       src = os.path.join(SOURCE, f)
       dst = os.path.join(TESTING, f)
       if os.path.getsize(src) > 0:
          copyfile(src, dst)  
#          print("copying: ", f)
       else:
          print("skipping empty file: ", f)

    
# YOUR CODE STARTS HERE
# YOUR CODE ENDS HERE
train_dir = "./training"
validation_dir = "./validation"

CAT_SOURCE_DIR = "./PetImages/Cat/"
TRAINING_CATS_DIR = "./training/cats/"
TESTING_CATS_DIR = "./validation/cats/"
DOG_SOURCE_DIR = "./PetImages/Dog/"
TRAINING_DOGS_DIR = "./training/dogs/"
TESTING_DOGS_DIR = "./validation/dogs/"


#
split_size = .9
#split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
#split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# Expected output
# 666.jpg is zero length, so ignoring
# 11702.jpg is zero length, so ignoring

print(len(os.listdir(TRAINING_CATS_DIR)))
print(len(os.listdir(TRAINING_DOGS_DIR)))
print(len(os.listdir(TESTING_CATS_DIR)))
print(len(os.listdir(TESTING_DOGS_DIR)))

# Expected output:
# 11250
# 11250
# 1250
# 1250


#sys.exit(0)

train_datagen = ImageDataGenerator(preprocessing_function=applications.resnet50.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=applications.resnet50.preprocess_input)

batch_size = 20

train_generator = train_datagen.flow_from_directory("./training", batch_size=batch_size, class_mode='binary', target_size=(224, 224))

validation_generator = test_datagen.flow_from_directory("./validation", batch_size=batch_size, class_mode='binary', target_size=(224, 224))

#model2.load_weights("transfer2.h5")

model2.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

history = model2.fit_generator(train_generator, 
                              validation_data=validation_generator, 
                              steps_per_epoch=100, 
                              epochs=5, 
                              validation_steps=50,
                              verbose=2)

model2.save_weights("transfer2.h5")



