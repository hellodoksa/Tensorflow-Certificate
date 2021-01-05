
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


#https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb
def original_horse_human() :
    #directory with our hourse pictures
    train_horse_dir = os.path.join('./tmp/horse-or-human/horses')
    train_human_dir = os.path.join('./tmp/horse-or-human/humans')

    ## look like in th horse and human
    train_horse_name = os.listdir(train_horse_dir)
    train_human_name = os.listdir(train_human_dir)
    # print(train_horse_name[:10])

    ## let's take a look at a few pictures
    '''
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4

    # Index for iterating over images
    pic_index = 0

    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_horse_pix = [os.path.join(train_horse_dir, fname)
                      for fname in train_horse_names[pic_index - 8:pic_index]]
    next_human_pix = [os.path.join(train_human_dir, fname)
                      for fname in train_human_names[pic_index - 8:pic_index]]

    for i, img_path in enumerate(next_horse_pix + next_human_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()
    '''

    # model maker
    '''
    Note that because we are facing a two-class classification problem, 
    i.e. a binary classification problem, 
    we will end our network with a sigmoid activation, so that the output of our network will be a single scalar between 0 and 1
    '''
    model = tf.keras.models.Sequential ([
        tf.keras.layers.Conv2D(16 , (3,3) , activation='relu' , input_shape=(300,300,3)),
        tf.keras.layers.MaxPool2D(2,2),

        tf.keras.layers.Conv2D(32, (3,3) , activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512 , activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    '''
    The convolution layers reduce the size of the feature maps by a bit due to padding, 
    and each pooling layer halves the dimensions.
    '''
    model.summary()


    '''
     In this case, using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD), 
     because RMSprop automates learning-rate tuning for us.
    '''
    from tensorflow.keras.optimizers import RMSprop
    model.compile(
        loss='binary_crossentropy' ,
        optimizer=RMSprop(lr=0.001) ,
        metrics=['accuracy']
    )

    # Data Processing
    '''
    data generators that will read pictures in our source folders, convert them to float32 tensors, 
    and feed them (with their labels) to our network.
    we will preprocess our images by normalizing the pixel values to be in the [0, 1] range
    '''
    from tensorflow.keras.preprocessing.image import ImageDataGenerator ## only tf2.4
    train_datagen = ImageDataGenerator(rescale=1/255)
    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        './tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    history = model.fit(
        train_generator ,
        steps_per_epoch= 8 ,
        epochs=15,
        verbose=1,
    )

    # how to convert
    path = './tmp/horse-or-human/horses/horse35-9.png'
    img = image.load_img(path, target_size=(300, 300))

    x = image.img_to_array(img) ## image to array
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(path + " is a human")
    else:
        print(path + " is a horse")

#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb
# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip  -O ./tmp/horse-or-human.zip
# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip -O /tmp/validation-horse-or-human.zip
def Validation () :
    import os
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb#scrollTo=ClebU9NJg99G
    # import zipfile
    # local_zip = './tmp/horse-or-human.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('./tmp/horse-or-human')
    # local_zip = './tmp/validation-horse-or-human.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('./tmp/validation-horse-or-human')
    # zip_ref.close()

    # Directory with our training horse pictures
    train_horse_dir = os.path.join('./tmp/horse-or-human/horses')
    # Directory with our training human pictures
    train_human_dir = os.path.join('./tmp/horse-or-human/humans')
    # Directory with our training horse pictures
    validation_horse_dir = os.path.join('./tmp/validation-horse-or-human/horses')
    # Directory with our training human pictures
    validation_human_dir = os.path.join('./tmp/validation-horse-or-human/humans')

    train_horse_names = os.listdir(train_horse_dir)
    print(train_horse_names[:10])

    train_human_names = os.listdir(train_human_dir)
    print(train_human_names[:10])

    validation_horse_hames = os.listdir(validation_horse_dir)
    print(validation_horse_hames[:10])

    validation_human_names = os.listdir(validation_human_dir)
    print(validation_human_names[:10])

    # model maker
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

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])

    # Validation !
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(
        './tmp/horse-or-human' ,
        target_size=(300,300) ,
        batch_size=128 ,
        class_mode='binary'
    )

    validation_generator = train_datagen.flow_from_directory(
        './tmp/validation-horse-or-human/' ,
        target_size=(300,300) ,
        batch_size=32 ,
        class_mode='binary'
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=8)


## exercise 
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > DESIRED_ACCURACY):
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True
        # Your Codedd

    callbacks = myCallback()

    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        # Your Code Here
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc']  # Your Code Here #)
                  )

    # This code block should create an instance of an ImageDataGenerator called train_datagen
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1 / 255)  # Your Code Here

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        '/tmp/h-or-s',
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary'
    )

    # Your Code Here)
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
        train_generator,
        epochs=15,
        callbacks=[callbacks],
        steps_per_epoch=8,
        verbose=1
    )

    # Your Code Here)
    # model fitting

    return history.history['acc'][-1]

if __name__ == '__main__' :
    # original_horse_human()
    Validation()
