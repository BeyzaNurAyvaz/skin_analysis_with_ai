# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 19:30:05 2022

@author: Ruzgar
"""
import tensorflow
from PIL import Image



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob




model_skin = tensorflow.keras.applications.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3)
                  )



model_skin.summary()


model_skin.trainable = True
set_trainable = False
for layer in model_skin.layers:
    if layer.name == 'block5_pool':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
        
        
  
        
  
model = tensorflow.keras.models.Sequential()   
model.add(model_skin)
model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
model.add(tensorflow.keras.layers.Dense(3, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])


train_path = "facial_datasets/TRAIN/"
test_path = "facial_datasets/TEST/"
validation_path = "facial_datasets/VALIDATION/"




#from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.3,
                                   horizontal_flip=True,
                                   zoom_range=0.3)



train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=32, #16
        )



validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

    
validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(224, 224),
        batch_size=16,
        )



trainModelSkin = model.fit(
      train_generator,
      steps_per_epoch=90, #10->90
      epochs=30, #5->25++
      validation_data=validation_generator,
      validation_steps=20)

model.save('model_skin.h5')

test_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=8,
        )

trainModelSkin.history

plt.plot (trainModelSkin.history['acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

























