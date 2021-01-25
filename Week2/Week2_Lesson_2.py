# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%202%20-%20Notebook%20(Cats%20v%20Dogs%20Augmentation).ipynb
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import zipfile
import tensorflow as tf
print(tf.__version__)
tf.config.list_physical_devices('GPU')


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

'''
[ DATA Augment ] 
These are just a few of the options available (for more, see the Keras documentation. Let's quickly go over what we just wrote:

rotation_range is a value in degrees (0–180), a range within which to randomly rotate pictures.
width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
shear_range is for randomly applying shearing transformations.
zoom_range is for randomly zooming inside pictures.
horizontal_flip is for randomly flipping half of the images horizontally. This is relevant when there are no assumptions of horizontal assymmetry (e.g. real-world pictures).
fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
'''


#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb
def augmentationPesonVSHorse() :
    # wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip -O ./tmp/validation-horse-or-human.zip
    # wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip -O ./tmp/horse-or-human.zip
    unzip = False
    if unzip :
        local_zip = './tmp/horse-or-human.zip'
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall('./tmp/horse-or-human')
        local_zip = './tmp/validation-horse-or-human.zip'
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall('./tmp/validation-horse-or-human')
        zip_ref.close()

    # Directory with our training horse pictures
    train_horse_dir = os.path.join('/tmp/horse-or-human/horses')

    # Directory with our training human pictures
    train_human_dir = os.path.join('/tmp/horse-or-human/humans')

    # Directory with our training horse pictures
    validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')

    # Directory with our training human pictures
    validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])



    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1/255 ,
        rotation_range=40,
        width_shift_range = 0.2,
        height_shift_range= 0.2,
        shear_range= 0.2, ## 영상 회전해서 피는거
        zoom_range= 0.2,
        horizontal_flip=True ,
        vertical_flip= False ,
        fill_mode='nearest' ## 빈 값은 주변 값으로 채운다.
    )

    validation_datagen = ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        '.\\Week2\\tmp\\horse-or-human\\',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    # Flow training images in batches of 128 using train_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
        '.\\Week2\\tmp\\validation-horse-or-human\\',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    history = model.fit (
        train_generator ,
        steps_per_epoch=8,
        epochs=20,
        verbose=1,
        validation_data= validation_generator ,
        validation_steps= 8
    )

    isshow = True
    if isshow :
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')

        plt.figure()

        plt.plot(epochs, loss, 'r', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()


#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%202%20-%20Notebook%20(Cats%20v%20Dogs%20Augmentation).ipynb
def augmentationCatVSDog() :
    #wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O ./tmp/cats_and_dogs_filtered.zip

    unzip = True
    if unzip :
        local_zip = './tmp/cats_and_dogs_filtered.zip'
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall('/tmp')
        zip_ref.close()

        base_dir = './tmp/cats_and_dogs_filtered'
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')

    # Directory with our training cat pictures
    train_cats_dir = os.path.join(train_dir, 'cats')

    # Directory with our training dog pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')

    # Directory with our validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')

    # Directory with our validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    # This code has changed. Now instead of the ImageGenerator just rescaling
    # the image, we also rotate and do other operations
    # Updated to do image augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    history = model.fit(
        train_generator,
        steps_per_epoch=100,  # 2000 images = batch_size * steps
        epochs=10,
        validation_data=validation_generator,
        validation_steps=50,  # 1000 images = batch_size * steps
        verbose=2)


def Exercise () :
    from shutil import copyfile
    from os import getcwd
    import shutil

    path_cats_and_dogs = f"{getcwd()}/../tmp2/cats-and-dogs.zip"
    shutil.rmtree('/tmp')

    local_zip = path_cats_and_dogs
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp')
    zip_ref.close()

    try:
        #     os.mkdir('/tmp')
        os.mkdir('/tmp/cats-v-dogs')
        os.mkdir('/tmp/cats-v-dogs/training')
        os.mkdir('/tmp/cats-v-dogs/testing')
        os.mkdir('/tmp/cats-v-dogs/training/cats')
        os.mkdir('/tmp/cats-v-dogs/training/dogs')
        os.mkdir('/tmp/cats-v-dogs/testing/cats')
        os.mkdir('/tmp/cats-v-dogs/testing/dogs')

        # YOUR CODE GOES HERE
    except Exception as e:
        print(e)
        pass

    # Write a python function called split_data which takes
    # a SOURCE directory containing the files
    # a TRAINING directory that a portion of the files will be copied to
    # a TESTING directory that a portion of the files will be copie to
    # a SPLIT SIZE to determine the portion
    # The files should also be randomized, so that the training set is a random
    # X% of the files, and the test set is the remaining files
    # SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
    # Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
    # and 10% of the images will be copied to the TESTING dir
    # Also -- All images should be checked, and if they have a zero file length,
    # they will not be copied over
    #
    # os.listdir(DIRECTORY) gives you a listing of the contents of that directory
    # os.path.getsize(PATH) gives you the size of the file
    # copyfile(source, destination) copies a file from source to destination
    # random.sample(list, len(list)) shuffles a list
    def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
        soure_img_list_num = len(os.listdir(SOURCE))
        train_num = int(soure_img_list_num * SPLIT_SIZE)
        test_num = int(soure_img_list_num * (1 - SPLIT_SIZE)) + 1

        train_random_img_list = random.sample(os.listdir(SOURCE), train_num)
        test_random_img_list = random.sample(os.listdir(SOURCE), test_num)
        print(len(train_random_img_list))
        print(len(test_random_img_list))

        for f in train_random_img_list:
            from_path = SOURCE + f
            copy_train_path = TRAINING + f
            copyfile(from_path, copy_train_path)

        for ff in test_random_img_list:
            copy_test_path = TESTING + ff
            copyfile(from_path, copy_test_path)

    # YOUR CODE STARTS HERE
    # YOUR CODE ENDS HERE

    CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
    TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
    TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
    DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
    TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
    TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

    split_size = .9
    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

    # DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
    # USE AT LEAST 3 CONVOLUTION LAYERS
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        # YOUR CODE HERE
    ])

    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

    TRAINING_DIR = '/tmp/cats-v-dogs/training/'
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')  # YOUR CODE HERE

    # NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE
    # TRAIN GENERATOR.
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=10,
        class_mode='binary',
        target_size=(150, 150)
    )  # YOUR CODE HERE

    VALIDATION_DIR = '/tmp/cats-v-dogs/testing/'  # YOUR CODE HERE
    validation_datagen = ImageDataGenerator(rescale=1 / 255)  # YOUR CODE HERE

    # NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE
    # VALIDATION GENERATOR.
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        batch_size=10,
        class_mode='binary',
        target_size=(150, 150)
    )  # YOUR CODE HERE

    # Expected Output:
    # Found 2700 images belonging to 2 classes.
    # Found 300 images belonging to 2 classes.

    history = model.fit_generator(train_generator,
                                  epochs=2,
                                  verbose=1,
                                  validation_data=validation_generator)
    # PLOT LOSS AND ACCURACY

    import matplotlib.image  as mpimg
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.figure()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")

    plt.title('Training and validation loss')

    # Desired output. Charts with training and validation metrics. No crash :)

if __name__ =='__main__' :
    augmentationPesonVSHorse()
    # augmentationCatVSDog()