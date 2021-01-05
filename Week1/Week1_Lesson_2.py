import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)


'''
https://www.tensorflow.org/tutorials/keras/classification
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=qnyTxjK_GbOD
'''
def basic() :
    mnist = tf.keras.datasets.fashion_mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()


    np.set_printoptions(linewidth=200)
    plt.imshow(training_images[0])
    print(training_labels[0])
    print(training_images[0])
    #
    training_images  = training_images / 255.0
    test_images = test_images / 255.0

    # print(training_images.shape)

    model = tf.keras.Sequential ( [
        tf.keras.layers.Flatten(input_shape=(28,28)) ,
        tf.keras.layers.Dense(128 , activation=tf.nn.relu),
        tf.keras.layers.Dense(10 , activation=tf.nn.softmax),
    ] )

    model.compile(
        optimizer=tf.optimizers.Adam() ,
        loss='sparse_categorical_crossentropy' ,
        metrics=['accuracy'] ,
    )

    model.fit(
        training_images,
        training_labels,
        epochs= 5,
    )

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\n테스트 정확도:', test_acc)

    classification = model.predict(test_images)
    print(np.argmax(classification[0]))


'''
In the course you learned how to do classificaiton using Fashion MNIST, a data set containing items of clothing. 
There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:
1. It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
3. If you add any additional variables, make sure you use the same names as the ones used in the class

I've started the code for you below -- how would you finish it? 
'''
def callback () : ## Exercise

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

    mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])


if __name__ == '__main__' :
    # basic()
    callback()
