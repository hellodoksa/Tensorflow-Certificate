# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb
'''
 - 처음 V3은 상단에 완전히 연결된 레이어를 가지고 있습니다. 따라서 include_top을 false로 설정하면이를 무시하고 회선으로 바로 이동하도록 지정합니다.
 - DropOut
 The idea behind Dropouts is that they remove a random number of neurons in your neural network.
 This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting,
 so dropping some out at random can remove this. The second is that often a neuron can over-weigh the input from a neuron in the previous layer,
 and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit!
'''
import os

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import os
import zipfile

def TransferModel () :
    # wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 -O ./tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    local_weights_file = './tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=None)

    pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    # pre_trained_model.summary()

    last_layer = pre_trained_model.get_layer('mixed10')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    ## This is not Sequential
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1 , activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)

    model.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(pre_trained_model.input)


    ## Data
    # wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O ./Week2/tmp/cats_and_dogs_filtered.zip
    unzip = True
    if unzip :
        local_zip = './tmp/cats_and_dogs_filtered.zip'

        zip_ref = zipfile.ZipFile(local_zip, 'r')

        zip_ref.extractall('./tmp')
        zip_ref.close()

    # Define our example directories and files
    base_dir = './tmp/cats_and_dogs_filtered'

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_cats_dir = os.path.join(train_dir, 'cats')  # Directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # Directory with our training dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # Directory with our validation cat pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # Directory with our validation dog pictures

    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)


    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=20,
                                                            class_mode='binary',
                                                            target_size=(150, 150))
    train_datagen.flow()
    history = model.fit_generator(  ## d=이걸 써야함
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_steps=50,
        verbose=2)

    show = False
    if show :

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        plt.figure()

        plt.show()

def Exercise () :
    from os import getcwd
    path_inception = f"{getcwd()}/../tmp2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

    # Import the inception model
    from tensorflow.keras.applications.inception_v3 import InceptionV3

    # Create an instance of the inception model from the local pre-trained weights
    local_weights_file = path_inception

    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=None)

    # Your Code Here

    pre_trained_model.load_weights(local_weights_file)

    # Make all the layers in the pre-trained model non-trainable
    for layer in pre_trained_model.layers:
        layer.trainable = False
    # Your Code Here

    # Print the model summary
    pre_trained_model.summary()

    # Expected Output is extremely large, but should end with:

    # batch_normalization_v1_281 (Bat (None, 3, 3, 192)    576         conv2d_281[0][0]
    # __________________________________________________________________________________________________
    # activation_273 (Activation)     (None, 3, 3, 320)    0           batch_normalization_v1_273[0][0]
    # __________________________________________________________________________________________________
    # mixed9_1 (Concatenate)          (None, 3, 3, 768)    0           activation_275[0][0]
    #                                                                 activation_276[0][0]
    # __________________________________________________________________________________________________
    # concatenate_5 (Concatenate)     (None, 3, 3, 768)    0           activation_279[0][0]
    #                                                                 activation_280[0][0]
    # __________________________________________________________________________________________________
    # activation_281 (Activation)     (None, 3, 3, 192)    0           batch_normalization_v1_281[0][0]
    # __________________________________________________________________________________________________
    # mixed10 (Concatenate)           (None, 3, 3, 2048)   0           activation_273[0][0]
    #                                                                 mixed9_1[0][0]
    #                                                                 concatenate_5[0][0]
    #                                                                 activation_281[0][0]
    # ==================================================================================================
    # Total params: 21,802,784
    # Trainable params: 0
    # Non-trainable params: 21,802,784

    last_layer = pre_trained_model.get_layer('mixed7')  # Your Code Here)
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output  # Your Code Here

    # Expected Output:
    # ('last layer output shape: ', (None, 7, 7, 768))

    from tensorflow.keras.optimizers import RMSprop

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)

    model.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy',  # Your Code Here,
                  metrics=['acc'])  # Your Code Here)

    model.summary()

    # Define a Callback class that stops training once accuracy reaches 97.0%
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.97):
                print("\nReached 97.0% accuracy so cancelling training!")
                self.model.stop_training = True

    # Get the Horse or Human dataset
    path_horse_or_human = f"{getcwd()}/../tmp2/horse-or-human.zip"
    # Get the Horse or Human Validation dataset
    path_validation_horse_or_human = f"{getcwd()}/../tmp2/validation-horse-or-human.zip"
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    import os
    import zipfile
    import shutil

    shutil.rmtree('/tmp')
    local_zip = path_horse_or_human
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp/training')
    zip_ref.close()

    local_zip = path_validation_horse_or_human
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp/validation')
    zip_ref.close()

    # Define our example directories and files
    train_dir = '/tmp/training'
    validation_dir = '/tmp/validation'

    train_horses_dir = os.path.join(train_dir, 'horses')  # Your Code Here
    train_humans_dir = os.path.join(train_dir, 'humans')  # Your Code Here
    validation_horses_dir = os.path.join(validation_dir, 'horses')  # Your Code Here
    validation_humans_dir = os.path.join(validation_dir, 'humans')  # Your Code Here

    train_horses_fnames = os.listdir(train_horses_dir)  # Your Code Here
    train_humans_fnames = os.listdir(train_humans_dir)  # Your Code Here
    validation_horses_fnames = os.listdir(validation_horses_dir)  # Your Code Here
    validation_humans_fnames = os.listdir(validation_humans_dir)  # Your Code Here

    # print(len(train_horses_fnames))  # Your Code Here)
    # print(len(train_humans_fnames))  # Your Code Here)
    # print(len(validation_horses_fnames))  # Your Code Here)
    # print(len(validation_humans_fnames))  # Your Code Here)

    # Expected Output:
    # 500
    # 527
    # 128
    # 128
    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    # Your Code Here)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=20,
                                                            class_mode='binary',
                                                            target_size=(150, 150))

    # Expected Output:
    # Found 1027 images belonging to 2 classes.
    # Found 256 images belonging to 2 classes.

    # Run this and see how many epochs it should take before the callback
    # fires, and stops training at 97% accuracy

    callbacks = myCallback()  # Your Code Here
    history = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=50,
        epochs=3,
        validation_steps=20,
        verbose=2,
        callbacks=[callbacks],
    )


import numpy as  np
np.array_split
if __name__ == '__main__' :
    TransferModel()