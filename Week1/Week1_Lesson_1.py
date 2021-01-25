import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def part2 () :
    model = keras.Sequential( [ keras.layers.Dense(units=1 , input_shape=[1])])
    model.compile(optimizer='sgd' , loss='mean_squared_error')

    xs = np.array( [-1.0 , 0.0, 1.0, 2.0, 3.0, 4.0] , dtype=float)
    ys = np.array( [-3.0 , -1.0, 1.0, 3.0, 5.0, 7.0] , dtype=float)

    model.fit(xs, ys, epochs=5000)

    print(model.predict( [10.0]))

#https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

def part3(exercise = 0) : # computer vision
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels) , (test_images, test_labels) = mnist.load_data()

    '''# show images
    import matplotlib.pyplot as plt
    plt.imshow(training_images[0])
    print(training_labels[0])
    print(training_images[0])
    '''

#it's easier if we treat all values as between 0 and 1, a process called 'normalizing'.
    training_images = training_images / 255.0
    test_images = test_images / 255.0

#design the model
#input size is different
    if exercise == 0 or 3 : #0.8429972
        print("@ Origianl test\nInput : 128")
        model = keras.models.Sequential( [
            keras.layers.Flatten(input_shape=(28, 28)) ,
            keras.layers.Dense(128 , activation=tf.nn.relu),
            keras.layers.Dense(10 , activation=tf.nn.softmax)
        ])

    #adding more Neurons we have to do more calculations, slowing down the process, but in this case they have a good impact
    elif exercise == 1:
        print("@ Input : 1024") ## 0.94817996
        model = keras.models.Sequential( [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    elif exercise == 2 :
        print("@ Add layer\nInput : 512")
        model = keras.models.Sequential( [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])


    #to actually build it.
    model.compile(
        optimizer=tf.optimizers.Adam() ,
        loss= 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    if exercise == 3:
        callbacks = myCallback()
        model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
    else:
        model.fit(training_images , training_labels, epochs= 5)

    print("@@ Evaluate")
    model.evaluate(test_images , test_labels)

    #@ Exercise 1
    ## it means It's the probability that this item is each of the 10 classes
    classifications = model.predict(test_images)
    print(classifications[0])
    print(classifications[0][-1])

    print(test_labels[0])

# Convolution!
'''
Add some layers to do convolution before you have the dense layers, 
and then the information going to the dense layers is more focussed, and possibly more accurate.
'''
def part4() :
    #@@ Have to check korean books for conv
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

## Reshape
    '''
    That's because the first convolution expects a single tensor containing everything,
    '''
    print(training_images.shape , test_images.shape)
    training_images = training_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1 )
    print(training_images.shape , test_images.shape)

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    model = keras.models.Sequential([
        keras.layers.Conv2D(128 , (3,3) , activation='relu', input_shape=(28,28,1)),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(64, (3,3) , activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(32, (3,3) , activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(128 , activation='relu'),
        keras.layers.Dense(10 , activation='softmax')
        ])

    model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(training_images, training_labels, epochs= 5)
    model.evaluate(test_images, test_labels)

def part5() :
    # !wget --no-check-certificate \
    #     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
    #     -O /tmp/horse-or-human.zip
    import os

    # for unzip
    # import zipfile
    # local_zip = './tmp/horse-or-human.zip'
    # zip_ref = zipfile.ZipFile(local_zip , 'r')
    # zip_ref.extractall('./tmp/horse-or-human')
    # zip_ref.close()

    #Let's define each of these directories:
    train_horse_dir = os.path.join('./tmp/horse-or-human/horses')
    train_human_dir = os.path.join('./tmp/horse-or-human/humans')

    train_horse_names = os.listdir(train_horse_dir)
    train_human_names = os.listdir(train_human_dir)

    # let's see what the filenames
    # print(train_horse_names[:10])
    # print(train_human_names[:10])

    # Let's find out the total number of horse
    print('total training horse images:', len(os.listdir(train_horse_dir)))
    print('total training human images:', len(os.listdir(train_human_dir)))

    show = False
    if show :
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

        test_img = mpimg.imread(os.path.join(train_horse_dir, train_horse_names[0]))
        print("1 image : " , test_img.shape)

#Building a Small Model from Scratch
    model = keras.models.Sequential ([
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

        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),

        # OutPut layer
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        # a binary classification problem, we will end our network with a sigmoid activation,
        tf.keras.layers.Dense(1, activation='sigmoid')

    ])
    model.summary()


    '''
    NOTE: In this case, using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD), 
    because RMSprop automates learning-rate tuning for us. 
    (Other optimizers, such as Adam and Adagrad, also automatically adapt the learning rate during training, and would work equally well here.)
    '''
    from tensorflow.keras.optimizers import RMSprop
    model.compile(
        optimizer=RMSprop(lr=0.001) ,
        loss= 'binary_crossentropy',
        metrics=['accuracy'],
    )

#Data Preprocessing
    '''
    It is uncommon to feed raw pixels into a convnet
    convert them to float32 tensors
    
     In our case, we will preprocess our images by normalizing the pixel values to be in the [0, 1] range 
     (originally all values are in the [0, 255] range).
    
    '''
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1 / 255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        './tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
    
    history = model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1)



    import random
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]
    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
    # Let's prepare a random input image from the training set.
    horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
    human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
    img_path = random.choice(horse_img_files + human_img_files)

    img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers[1:]]

    # Now let's display our representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            n_features = feature_map.shape[-1]  # number of features in feature map
            # The feature map has shape (1, size, size, n_features)
            size = feature_map.shape[1]
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                # Postprocess the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into this big horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x
            # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

    # import os, signal
    # os.kill(os.getpid(), signal.SIGKILL)

if __name__ == '__main__' :
    # part2()
    # part3()
    # part4()
    part5()
np.ar