'''
https://www.tensorflow.org/tutorials

'''
import  tensorflow as tf
print(tf.__version__)
import tensorflow_datasets as tfds
import numpy as np
import os
import PIL
import PIL.Image

#https://www.tensorflow.org/tutorials/load_data/images
def FlowerClassification () :
    '''
    Data Set
    flowers_photos/
      daisy/
      dandelion/
      roses/
      sunflowers/
      tulips/
    '''
    import pathlib
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                       fname='flower_photos',
                                       untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count , "hello")

    roses = list(data_dir.glob('roses/*'))
    # PIL.Image.open(str(roses[0]))

    ## Make Data Set for Validation, Test
    batch_size = 12 ; img_width = 180 ; img_height = 180 ;

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(


    )


if __name__ == '__main__' :
    FlowerClassification ()