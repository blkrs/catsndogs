from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import numpy as np


img_shape = (224, 224, 3)
img_input = Input(shape=img_shape)

model = ResNet50(weights='imagenet', input_tensor=img_input)

flat = Flatten()(model.layers[-3].output)
dense = Dense(64, activation='relu')(flat)
dense2 = Dense(32, activation='relu')(dense)
dogcat = Dense(1, activation="sigmoid")(dense2)


img_path = 'krzych.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(model.layers[0])
#in_img = model.layers[0].inputs
for layer in model.layers:
    layer.trainable=False

model2 = Model(inputs=[img_input], outputs=[dogcat])

preds = model2.predict(x)

print("Predictions : ", preds)

model2.summary()
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
