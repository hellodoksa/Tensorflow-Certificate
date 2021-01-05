import tensorflow as tf

def DenseFunc () :
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accura    cy'])
    model.fit(training_images, training_labels, epochs=5)

    test_loss = model.evaluate(test_images, test_labels)
def ConvFunc () :
    print(tf.__version__)
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(training_images, training_labels, epochs=5)
    test_loss = model.evaluate(test_images, test_labels)

def FilterExperiment () :
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb
# https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb
    import numpy as np
    from scipy import misc
    i = misc.ascent()

def MiniExam () :
## https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb
    # GRADED FUNCTION: train_mnist_conv
    # def train_mnist_conv():
        # Please write your code only where you are indicated.
        # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    training_images = training_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)

    callbacks = myCallback()

    training_images = training_images / 255.0
    test_images = test_images / 255.0
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
        # YOUR CODE STARTS HERE
        tf.keras.layers.Conv2D(14, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(14, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
        training_images, training_labels,
        epochs=20,
        callbacks=[callbacks],

        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]


if __name__ == '__main__' :
#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=8KVPZqgHo5Ux
    '''EXERCISES

1. Try editing the convolutions. Change the 32s to either 16 or 64. What impact will this have on accuracy and/or training time.
    - 1st try (64, 64) : 313/313 [==============================] - 1s 3ms/step - loss: 0.2548 - accuracy: 0.9073
    - 2nd try (32, 32) : 313/313 [==============================] - 1s 2ms/step - loss: 0.2707 - accuracy: 0.9007
    - 3rd try (16, 16) : 313/313 [==============================] - 1s 2ms/step - loss: 0.2977 - accuracy: 0.8893
    - 4th try (64, 32) : 313/313 [==============================] - 1s 3ms/step - loss: 0.2625 - accuracy: 0.9040
2. Remove the final Convolution. What impact will this have on accuracy or training time?

3. How about adding more Convolutions? What impact do you think this will have? Experiment with it.

4. Remove all Convolutions but the first. What impact do you think this will have? Experiment with it.

5. In the previous lesson you implemented a callback to check on the loss function and to cancel training once it hit a certain amount. See if you can implement that here! 
    '''
    # DenseFunc()
    # ConvFunc()
    # FilterExperiment