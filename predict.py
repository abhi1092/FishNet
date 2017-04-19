from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import load_model
import pandas as pd
import csv
import numpy

img_width, img_height = 150, 150
validation_data_dir = 'data/test'
nb_validation_samples = 16

batch_size = 16

model = applications.VGG16(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)



model = load_model('fc_model.h5')

classes = model.predict(bottleneck_features_validation)
print(classes.shape)

y = [x.split("\\")[1] for x in generator.filenames]


image_list = pd.DataFrame(y,columns=['image'])
probability = pd.DataFrame(classes,columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF','OTHER','SHARK','YFT'])
fina_sub = pd.concat([image_list, probability], axis=1)
fina_sub.to_csv("output.csv",sep=',')
