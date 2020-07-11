import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras import optimizers
import keras

now_path=os.getcwd()
loacl_path=[]
for i in range(len(now_path)):
    loacl_path.append(now_path[i])
for j in range(len(loacl_path)):
    if loacl_path[j]=='\\':
        loacl_path[j]='/'
now_path="".join(loacl_path)
train_dir=now_path+'/covid/train/'


img_width, img_height = 299, 299
train_size = 5536


conv_base = keras.applications.InceptionV3(weights='imagenet',include_top = False,input_shape=(img_width, img_height, 3))
conv=conv_base.output.shape

conv_base.summary()
datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 16
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, conv[1], conv[2], conv[3]))
    labels = np.zeros(shape=(sample_count,4))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='categorical')

    i = 0
    labels_name = generator.class_indices
    labels_name = dict((v, k) for k, v in labels_name.items())
    print(labels_name)
    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels
train_features, train_labels = extract_features(train_dir, train_size)

epochs = 50
model = models.Sequential()
model.add(layers.Flatten(input_shape=train_features.shape[1:]))
model.add(layers.Dense(256, activation='relu', input_dim=(train_features.shape[1]*train_features.shape[2]*train_features.shape[3])))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(4, activation='softmax'))
model.summary()
model.compile(optimizer=optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['acc'])
history = model.fit(train_features, train_labels,batch_size=batch_size,epochs=epochs)
model.save('./donghwi12_COVID19')

