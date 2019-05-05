# In this exercise you will train a CNN on the FULL Cats-v-dogs dataset
# This will require you doing a lot of data preprocessing because
# the dataset isn't split into training and validation for you
# This code block has all the required inputs
import os
import sys
import zipfile
import random
import tensorflow as tf
import time
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# This code block downloads the full Cats-v-Dogs dataset and stores it as 
# cats-and-dogs.zip. It then unzips it to /tmp
# which will create a tmp/PetImages directory containing subdirectories
# called 'Cat' and 'Dog' (that's how the original researchers structured it)
# If the URL doesn't work, 
# .   visit https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# And right click on the 'Download Manually' link to get a new URL

#!wget --no-check-certificate \
#    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
#    -O "/tmp/cats-and-dogs.zip"

#local_zip = '/tmp/cats-and-dogs.zip'
#zip_ref = zipfile.ZipFile(local_zip, 'r')
#zip_ref.extractall('/tmp')
#zip_ref.close()


def visualize():
        import numpy as np
        import random
        from   tensorflow.keras.preprocessing.image import img_to_array, load_img
        import matplotlib
        matplotlib.use("PDF")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        # Directory with our training cat/dog pictures
        train_cats_dir = os.path.join(train_dir, 'cats')
        train_dogs_dir = os.path.join(train_dir, 'dogs')

        # Directory with our validation cat/dog pictures
        validation_cats_dir = os.path.join(validation_dir, 'cats')
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')


        train_cat_fnames = os.listdir( train_cats_dir )
        train_dog_fnames = os.listdir( train_dogs_dir )
        # Let's define a new Model that will take an image as input, and will output
        # intermediate representations for all layers in the previous model after
        # the first.
        successive_outputs = [layer.output for layer in model.layers[1:]]

        #visualization_model = Model(img_input, successive_outputs)
        visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

        # Let's prepare a random input image of a cat or dog from the training set.
        cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
        dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]

        img_path = random.choice(cat_img_files + dog_img_files)
        img = load_img(img_path, target_size=(150, 150))  # this is a PIL image

        x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
        x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

        # Rescale by 1/255
        x /= 255.0

        # Let's run our image through our network, thus obtaining all
        # intermediate representations for this image.
        successive_feature_maps = visualization_model.predict(x)

        # These are the names of the layers, so can have them as part of our plot
        layer_names = [layer.name for layer in model.layers]

        # -----------------------------------------------------------------------
        # Now let's display our representations
        # -----------------------------------------------------------------------
        for layer_name, feature_map in zip(layer_names, successive_feature_maps):
          
          if len(feature_map.shape) == 4:
            
            #-------------------------------------------
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            #-------------------------------------------
            n_features = feature_map.shape[-1]  # number of features in the feature map
            size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
            
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            
            #-------------------------------------------------
            # Postprocess the feature to be visually palatable
            #-------------------------------------------------
            for i in range(n_features):
              x  = feature_map[0, :, :, i]
              x -= x.mean()
              x /= x.std ()
              x *=  64
              x += 128
              x  = np.clip(x, 0, 255).astype('uint8')
              display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

            #-----------------
            # Display the grid
            #-----------------

            scale = 20. / n_features
            plt.figure( figsize=(scale * n_features, scale) )
            plt.title ( layer_name )
            plt.grid  ( False )
            print("showing img:")
            plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 
            ts = time.time()
        
        plt.savefig("plots/plot_"+str(ts)+".png")



print(len(os.listdir('./PetImages/Cat/')))
print(len(os.listdir('./PetImages/Dog/')))

# Expected Output:
# 12501
# 12501

# Use os.mkdir to create your directories
# You will need a directory for cats-v-dogs, and subdirectories for training
# and testing. These in turn will need subdirectories for 'cats' and 'dogs'
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


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

model.summary()

model.load_weights("model.h5")
visualize()

#sys.exit(0)

train_datagen = ImageDataGenerator( rescale = 1.0/255., rotation_range = 30 )
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

batch_size = 20

train_generator = train_datagen.flow_from_directory("./training", batch_size=batch_size, class_mode='binary', target_size=(150, 150))

validation_generator = test_datagen.flow_from_directory("./validation", batch_size=batch_size, class_mode='binary', target_size=(150, 150))

history = model.fit_generator(train_generator, 
                              validation_data=validation_generator, 
                              steps_per_epoch=100, 
                              epochs=10, 
                              validation_steps=50,
                              verbose=2)

if history.history['val_acc'][-1] > history.history['val_acc'][0]:
    model.save_weights("model.h5")



