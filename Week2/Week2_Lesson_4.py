# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import keras_preprocessing
from keras_preprocessing import image


def rsp () :
    # wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip -O ./tmp/rps.zip
    # wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip -O ./tmp/rps-test-set.zip

    unzip = False
    if unzip :
        local_zip = './tmp/rps.zip'
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall('./tmp/')
        zip_ref.close()

        local_zip = './tmp/rps-test-set.zip'
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall('./tmp/')
        zip_ref.close()

    rock_dir = os.path.join('./tmp/rps/rock')
    paper_dir = os.path.join('./tmp/rps/paper')
    scissors_dir = os.path.join('./tmp/rps/scissors')

    print('total training rock images:', len(os.listdir(rock_dir)))
    print('total training paper images:', len(os.listdir(paper_dir)))
    print('total training scissors images:', len(os.listdir(scissors_dir)))

    rock_files = os.listdir(rock_dir)
    print(rock_files[:10])

    paper_files = os.listdir(paper_dir)
    print(paper_files[:10])

    scissors_files = os.listdir(scissors_dir)
    print(scissors_files[:10])

    draw = False
    if draw :
        pic_index = 2

        next_rock = [os.path.join(rock_dir, fname)
                     for fname in rock_files[pic_index - 2:pic_index]]
        next_paper = [os.path.join(paper_dir, fname)
                      for fname in paper_files[pic_index - 2:pic_index]]
        next_scissors = [os.path.join(scissors_dir, fname)
                         for fname in scissors_files[pic_index - 2:pic_index]]

        for i, img_path in enumerate(next_rock + next_paper + next_scissors):
            # print(img_path)
            img = mpimg.imread(img_path)
            plt.imshow(img)
            plt.axis('Off')
            plt.show()


    TRAINING_DIR = "./tmp/rps/"
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    VALIDATION_DIR = "./tmp/rps-test-set/"
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=126
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=126
    )

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data=validation_generator, verbose=1,
                        validation_steps=3)

    model.save("rps.h5")
def exercise () :
    def get_data(filename):
        # You will need to write code that will read the file passed
        # into this function. The first line contains the column headers
        # so you should ignore it
        # Each successive line contians 785 comma separated values between 0 and 255
        # The first value is the label
        # The rest are the pixel values for that picture
        # The function will return 2 np.array types. One with all the labels
        # One with all the images
        #
        # Tips:
        # If you read a full line (as 'row') then row[0] has the label
        # and row[1:785] has the 784 pixel values
        # Take a look at np.array_split to turn the 784 pixels into 28x28
        # You are reading in strings, but need the values to be floats
        # Check out np.array().astype for a conversion
        limage = [];
        llabel = []
        with open(filename) as training_file:
            all_files = csv.reader(training_file)
            next(all_files)
            for row in all_files:
                labels = row[0]
                data = row[1:]
                img = np.array(data).reshape((28, 28))

                limage.append(img)
                llabel.append(labels)

            images = np.array(limage).astype(float)
            labels = np.array(llabel).astype(float)

        # Your code starts here
        # Your code ends here
        return images, labels

    path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
    path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
    training_images, training_labels = get_data(path_sign_mnist_train)
    testing_images, testing_labels = get_data(path_sign_mnist_test)

    # Keep these

    print(training_images.shape)
    print(training_labels.shape)
    print(testing_images.shape)
    print(testing_labels.shape)

    # In this section you will have to add another dimension to the data
    # So, for example, if your array is (10000, 28, 28)
    # You will need to make it (10000, 28, 28, 1)
    # Hint: np.expand_dims

    training_images = np.expand_dims(training_images, 3)
    # Your Code Here
    testing_images = np.expand_dims(testing_images, 3)  # Your Code Here

    # Create an ImageDataGenerator and do Image Augmentation
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    # Keep These
    print(training_images.shape)
    print(testing_images.shape)


    # @@@@@@@@@@@
    train_generator = train_datagen.flow(
        training_images, training_labels, batch_size=85
    )

    # Define the model
    # Use no more than 2 Conv2D and 2 MaxPooling2D
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(25, activation='softmax')
    ])

    # Compile Model.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Train the Model
    # Your Code Here (set 'epochs' = 2))

    '''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@222
    '''
    # history = model.fit_generator(training_images , training_labels, epochs = 2 ,steps_per_epoch=None , verbose=0)
    history = model.fit_generator(
        train_datagen.flow(training_images, training_labels, batch_size=85),
        steps_per_epoch=27455 // 85,
        epochs=2,
        validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),
        validation_steps=7172 // 32,
        verbose=0,
    )

    model.evaluate(testing_images, testing_labels, verbose=0)

if __name__ == '__main__' :
    rsp()
