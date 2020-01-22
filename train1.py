import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense,Dropout




from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical') 
print(training_set.class_indices)
print(test_set.class_indices)
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
checkpoint = ModelCheckpoint('vgg16_mod.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='auto')  
red=ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=5,mode="auto")
early=EarlyStopping(monitor="val_loss",min_delta=1e-4,patience=10,mode="auto")
from tensorflow.keras.models import load_model
model = load_model('agerta.h5')
model.compile(optimizer ='adam', loss="categorical_crossentropy", metrics=["accuracy"])
'''
model.fit_generator(
        training_set,
        steps_per_epoch=1000,
        epochs=80,
        validation_data=test_set,
        validation_steps=500,
        callbacks=[checkpoint,red,early])

'''

model.save("model2.h5")
