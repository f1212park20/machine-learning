import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

# !head -n 2 /kaggle/input/digit-recognizer/train.csv
train_data = pd.read_csv("{}/train.csv".format(dirname))
test_data = pd.read_csv("{}/test.csv".format(dirname))

train_data.head()

test_data.to_numpy().shape

train_data.to_numpy().shape

np.max(train_data.to_numpy()[0])

train_label = train_data.label.to_numpy()
train_image = train_data.to_numpy()[0:, 1:].reshape(42000, 28, 28, 1)
test_image = test_data.to_numpy().reshape(28000, 28, 28, 1)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img, cmap="gray")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


datagen = ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # horizontal_flip=True,
    validation_split=0.2)

train_data_gen = datagen.flow(train_image, train_label, batch_size=294, shuffle=True, subset="training")
valid_data_gen = datagen.flow(train_image, train_label,
                              batch_size=126, subset="validation")

augmented_images = [train_data_gen[0][0][i] for i in range(5)]
print(train_data_gen[0][1][0:5])
plotImages(augmented_images)

import keras_tuner as kt
import IPython

print("shape:", train_data_gen[0][0][0].shape)
print("max:", np.max(train_data_gen[0][0][0]))

train_image[0].shape


def model_builder(hp):
    # input layer
    hp_input_layer = hp.Int("InputParam", min_value=28, max_value=28, step=4)
    # layer1
    hp_drop_rate1 = hp.Choice('drop_rate1', values=[0.2])
    hp_layer_units1 = hp.Int('units1', min_value=28, max_value=28, step=4)
    hp_reg_rate1 = hp.Choice('reg_rate1', values=[1e-4])
    # layer2
    hp_layer_units2 = hp.Int('units2', min_value=32, max_value=36, step=4)
    hp_reg_rate2 = hp.Choice('reg_rate2', values=[1e-4])

    hp_layer3_flag = hp.Choice('layer3_flag', values=[True, False])
    # layer3
    hp_layer_units3 = hp.Int('units3', min_value=24, max_value=32, step=4)
    hp_reg_rate3 = hp.Choice('reg_rate3', values=[1e-4])

    model = keras.Sequential([

        Conv2D(hp_input_layer, (3, 3), activation='relu',
               input_shape=train_image[0].shape),
        MaxPooling2D(),
        # layer1
        Conv2D(hp_layer_units1, (3, 3), activation='relu',
               kernel_regularizer=keras.regularizers.l2(hp_reg_rate1)),
        MaxPooling2D(),
        keras.layers.Dropout(hp_drop_rate1),

        # layer2
        Conv2D(hp_layer_units2, (3, 3), activation='relu',
               kernel_regularizer=keras.regularizers.l2(hp_reg_rate2)),

    ])

    if hp_layer3_flag == True:
        model.add(
            Conv2D(hp_layer_units3, (3, 3), activation='relu',
                   kernel_regularizer=keras.regularizers.l2(hp_reg_rate3))
        )

    model.add(
        Flatten())
    model.add(Dense(10, activation='softmax'))

    # compile
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=["accuracy", ])
    return model


tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=50,
                     directory='my_dir',
                     project_name='intro_to_kt')


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


tuner.search(train_data_gen,
             steps_per_epoch=100,
             epochs=50,
             validation_data=valid_data_gen,
             validation_steps=50,
             callbacks=[ClearTrainingOutput(), early_stop]
             )


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
model.summary()


hist = model.fit(train_data_gen,
                 steps_per_epoch=100,
                 epochs=200,
                 validation_data=valid_data_gen,
                 validation_steps=50,
                 callbacks=[ClearTrainingOutput(), early_stop],
                 verbose=0
                 )


history_dict = hist.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)


plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

res = model.predict(test_image[:] / 255)
output = pd.DataFrame({'ImageId': [i + 1 for i in range(len(test_image))],
                       'Label': [xi.argmax() for xi in res]})
output.to_csv('submission_grid.csv', index=False)

for i in range(5):
    print("predicted: ", res[i].argmax())
    plt.imshow(test_image[i], cmap="gray")
    plt.show()
