# import all necessary libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Activation, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# input pyplot for plotting
from matplotlib import pyplot as plt

def main():
    
    epochs = 10
    batch_size = 16
    # desired img size after resizing
    img_width = 200
    img_height = 200

    training_validation_data_dir = 'plant-seedlings-classification/train'

    # model design
    model = Sequential()

    # input layer
    model_input_layer = InputLayer(
        input_shape = (img_width, img_height, 3)
    )
    model.add(model_input_layer)

    # CNN layer using InceptionResNetV2
    model_CNN_layer = InceptionResNetV2(
        include_top= False
    )
    model.add(model_CNN_layer)

    # Flatten the img and add dense layer
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    # classify 12 catagories
    model.add(Dense(12))
    model.add(Activation('softmax'))

    # use Adam as optimizer
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer= adam_optimizer, 
        metrics=["accuracy"]
    )

    # output the summary of the model to check if the model structure
    model.summary()

    # Hyperparameter tuning based on the img in dataset
    train_datagen = ImageDataGenerator(
        rotation_range = 180,
        width_shift_range = 0.3,
        height_shift_range = 0.3,
        shear_range = 0.3,
        zoom_range = 0.5,
        vertical_flip = True,
        horizontal_flip = True,
        # split the dataset to training subset - 80% and validation subset - 20%
        validation_split = 0.2
    )

    train_generator = train_datagen.flow_from_directory(
        directory = training_validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = "categorical",
        subset = 'training',
    )

    validation_generator = train_datagen.flow_from_directory(
        directory = training_validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = "categorical",
        subset = 'validation',
    )

    training_history = model.fit(
		train_generator,
		steps_per_epoch=train_generator.samples // batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=validation_generator.samples // batch_size
    )

    model.save('plant_seedlings_classification.keras')

    # plot accuracy over epoch graph
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.xlabel('epoch [times]')
    plt.ylabel('accuracy [rate]')
    plt.legend(['training','validation'], loc='upper left')
    plt.savefig('accuracy.png')



if __name__ == "__main__":
    main()