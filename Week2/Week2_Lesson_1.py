# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=mu3Jdwkjwax4
# https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O ./Week2/tmp/cats_and_dogs_filtered.zip
def CatAndDog () :
    unzip = False
    if unzip :
        local_zip = './tmp/cats_and_dogs_filtered.zip'
        zip_ref = zipfile.ZipFile(local_zip , 'r')

        zip_ref.extractall('./tmp')
        zip_ref.close()

    base_dir = './tmp/cats_and_dogs_filtered'

    ## File Lists
    train_dir       = os.path.join( base_dir , 'train')
    validation_dir  = os.path.join( base_dir , 'validation')

    # Directory with our training cat/dog pictures
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')

    # Directory with our validation cat/dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    train_cat_fnames = os.listdir (train_cats_dir)
    train_dog_fnames = os.listdir (train_dogs_dir)

    print ( len(train_cat_fnames) , train_cat_fnames[:5])
    print ( len(train_cat_fnames) , train_cat_fnames[:5])


    isshow = False
    if isshow :

        # Parameters for our graph; we'll output images in a 4x4 configuration
        nrows = 4
        ncols = 4

        pic_index = 0  # Index for iterating over images

        # Set up matplotlib fig, and size it to fit 4x4 pics
        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)

        pic_index += 8

        next_cat_pix = [os.path.join(train_cats_dir, fname)
                        for fname in train_cat_fnames[pic_index - 8:pic_index]
                        ]

        next_dog_pix = [os.path.join(train_dogs_dir, fname)
                        for fname in train_dog_fnames[pic_index - 8:pic_index]
                        ]

        for i, img_path in enumerate(next_cat_pix + next_dog_pix):
            # Set up subplot; subplot indices start at 1
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off')  # Don't show axes (or gridlines)

            img = mpimg.imread(img_path)
            plt.imshow(img)

        plt.show()

    model = tf.keras.models.Sequential ([
        tf.keras.layers.Conv2D ( 16 , (3,3) , activation= 'relu' , input_shape= (150, 150, 3)) ,
        tf.keras.layers.MaxPool2D(2,2),

        tf.keras.layers.Conv2D(32 , (3,3) , activation='relu'),
        tf.keras.layers.MaxPool2D(2,2) ,

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Flatten() ,
        tf.keras.layers.Dense(512 , activation='relu') ,
        tf.keras.layers.Dense(1 , activation='sigmoid')
    ])

    model.summary()
    model.compile(
        loss = 'binary_crossentropy', #tf.keras.losses.binary_crossentropy,
        optimizer= tf.keras.optimizers.RMSprop (lr = 0.001) ,
        metrics=['accuracy']
    )

    # DATA PREPROCESSING  -> generator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # All images will be rescaled by 1./255.
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    # --------------------
    # Flow training images in batches of 20 using train_datagen generator
    # --------------------
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))
    # --------------------

    # Flow validation images in batches of 20 using test_datagen generator
    # --------------------
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=20,
                                                            class_mode='binary',
                                                            target_size=(150, 150))

    ## TRAING
    ## Let's train on all 2,000 images available, for 15 epochs, and validate on all 1,000 test images
    history = model.fit (
        train_generator ,
        validation_data= validation_generator ,
        steps_per_epoch= 100 ,
        epochs= 15 ,
        validation_steps= 50 ,
        verbose= 2
    )

    ## Prediction

    img_path  = './tmp/test_cat1.jpg'
    opencv = True
    if opencv :
        import cv2
        x= cv2.imread(img_path)
        x = cv2.resize(x , dsize=(150,150))


    elif not opencv  :
        img = image.load_img(img_path)
        x = image.img_to_array(img) ##KERAS

    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])


    classes = model.predict(images, batch_size=10)
    print(classes[0])

    if classes[0] > 0:
        print(img_path + " is a dog")

    else:
        print(img_path + " is a cat")

    ## Evaluationg using 'history'
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    print(f"Evaluationg epochs {epochs}")

    ## plot
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.figure()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    plt.show()

def exercise () :
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
    pass
# os.listdir(DIRECTORY) gives you a listing of the contents of that directory
# os.path.getsize(PATH) gives you the size of the file
# copyfile(source, destination) copies a file from source to destination
# random.sample(list, len(list)) shuffles a list
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    soure_img_list = os.listdir(SOURCE)
if __name__ == '__main__' :
    CatAndDog()
