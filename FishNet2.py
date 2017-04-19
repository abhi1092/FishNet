import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 3056
nb_validation_samples = 736
epochs = 70
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy','rb'))

    train_labels = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0]] * (1363) + [[0,1,0,0,0,0,0,0]] * (200) + [[0,0,1,0,0,0,0,0]] * (93) + [[0,0,0,1,0,0,0,0]] * (54) + [[0,0,0,0,1,0,0,0]] * (375) + [[0,0,0,0,0,1,0,0]] * (239) + [[0,0,0,0,0,0,1,0]] * (141) + [[0,0,0,0,0,0,0,1]] * (591))

    validation_data = np.load(open('bottleneck_features_validation.npy','rb'))

    validation_labels = np.array(
        [[1,0,0,0,0,0,0,0]] * (331) + [[0,1,0,0,0,0,0,0]] * (40) + [[0,0,1,0,0,0,0,0]] * (24) + [[0,0,0,1,0,0,0,0]] * (13) + [[0,0,0,0,1,0,0,0]] * (90) + [[0,0,0,0,0,1,0,0]] * (60) + [[0,0,0,0,0,0,1,0]] * (35) + [[0,0,0,0,0,0,0,1]] * (143))
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    model.save('fc_model.h5')



save_bottlebeck_features()
train_top_model()