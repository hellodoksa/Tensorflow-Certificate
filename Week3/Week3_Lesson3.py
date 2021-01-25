from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)
def SwitchModel ():
    # https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202d.ipynb#scrollTo=nHGYuU4jPYaj
    # https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202d.ipynb#scrollTo=K_Jc7cY3Qxke

    '''
    Using imdb_reviews DataSet

    ################
     DATA Preprocessing
    ################'''
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']

    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    # str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
    for s, l in train_data:
        training_sentences.append(str(s.numpy()))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(str(s.numpy()))
        testing_labels.append(l.numpy())

    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    '''################
    Tokenizer
    ################'''
    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    print(decode_review(padded[1]))
    print(training_sentences[1])

    '''################
        Model Select 
    ################'''
    which_model = 4

    # ''' Flatten '''
    if which_model == 1 :
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Flatten() ,
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        '''
        Model: "sequential_3"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        embedding_3 (Embedding)      (None, 120, 16)           160000    
        _________________________________________________________________
        flatten (Flatten)            (None, 1920)              0         
        _________________________________________________________________
        dense_6 (Dense)              (None, 6)                 11526     
        _________________________________________________________________
        dense_7 (Dense)              (None, 1)                 7         
        =================================================================
        Total params: 171,533
        Trainable params: 171,533
        Non-trainable params: 0
        _________________________________________________________________
        '''

    # ''' GRU  ''' => Total params: 169,997
    elif which_model == 2 :
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        '''
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        embedding (Embedding)        (None, 120, 16)           160000    
        _________________________________________________________________
        bidirectional (Bidirectional (None, 64)                9600      
        _________________________________________________________________
        dense (Dense)                (None, 6)                 390       
        _________________________________________________________________
        dense_1 (Dense)              (None, 1)                 7         
        =================================================================
        Total params: 169,997
        Trainable params: 169,997
        Non-trainable params: 0
        _________________________________________________________________
        '''

    # ''' Conv1D  ''' => Total params: 171,149
    elif which_model == 3 :
        # Model Definition with Conv1D
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        '''
        Model: "sequential_2"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        embedding_2 (Embedding)      (None, 120, 16)           160000    
        _________________________________________________________________
        conv1d (Conv1D)              (None, 116, 128)          10368     
        _________________________________________________________________
        global_average_pooling1d (Gl (None, 128)               0         
        _________________________________________________________________
        dense_4 (Dense)              (None, 6)                 774       
        _________________________________________________________________
        dense_5 (Dense)              (None, 1)                 7         
        =================================================================
        Total params: 171,149
        Trainable params: 171,149
        Non-trainable params: 0
        _________________________________________________________________
        '''

    # ''' LSTM ''' => Total params: 172,941
    elif which_model == 4 :
        # Model Definition with LSTM
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        '''
        Model: "sequential_1"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        embedding_1 (Embedding)      (None, 120, 16)           160000    
        _________________________________________________________________
        bidirectional_1 (Bidirection (None, 64)                12544     
        _________________________________________________________________
        dense_2 (Dense)              (None, 6)                 390       
        _________________________________________________________________
        dense_3 (Dense)              (None, 1)                 7         
        =================================================================
        Total params: 172,941
        Trainable params: 172,941
        Non-trainable params: 0
        _________________________________________________________________
        '''
    num_epochs = 50
    history = model.fit(padded, training_labels_final, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels_final))
    import matplotlib.pyplot as plt

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')

def SwitchModel_sarcasm() :
    # https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202.ipynb#scrollTo=jGwXGIXvFhXW
    # https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202c.ipynb#scrollTo=7ZEZIUppGhdi
    import json
    ''' 선언  '''
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    ''' Data Preprocessing '''
    with open("./tmp/sarcasm.json", 'r') as f:
        datastore = json.load(f)

    sentences = []
    labels = []
    urls = []
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    ''' for train '''
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    ''' Model Select '''
    which_model = 2

    # LSTM
    if which_model == 1 :
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        '''
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        embedding (Embedding)        (None, 120, 16)           16000     
        _________________________________________________________________
        bidirectional (Bidirectional (None, 64)                12544     
        _________________________________________________________________
        dense (Dense)                (None, 24)                1560      
        _________________________________________________________________
        dense_1 (Dense)              (None, 1)                 25        
        =================================================================
        Total params: 30,129
        Trainable params: 30,129
        Non-trainable params: 0
        _________________________________________________________________
        '''

    ## Conv
    elif which_model == 2 :
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        '''
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        embedding (Embedding)        (None, 120, 16)           16000     
        _________________________________________________________________
        conv1d (Conv1D)              (None, 116, 128)          10368     
        _________________________________________________________________
        global_max_pooling1d (Global (None, 128)               0         
        _________________________________________________________________
        dense (Dense)                (None, 24)                3096      
        _________________________________________________________________
        dense_1 (Dense)              (None, 1)                 25        
        =================================================================
        Total params: 29,489
        Trainable params: 29,489
        Non-trainable params: 0
        _________________________________________________________________        
        '''
    num_epochs = 50
    history = model.fit(training_padded, training_labels, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels), verbose=1)
    import matplotlib.pyplot as plt

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
    model.save("test.h5")

''' 
When looking at a number of different types of layer for text classification this week you saw many examples of overfitting 
-- with one of the major reasons for the overfitting being that your training dataset was quite small, 
and with a small number of words. Embeddings derived from this may be over generalized also. 
So for this week’s exercise you’re going to train on a large dataset, as well as using transfer learning of an existing set of embeddings.

#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP%20Course%20-%20Week%203%20Exercise%20Answer.ipynb#scrollTo=ohOGz24lsNAD
'''
def Exercise_Large_Dataset() :
#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP%20Course%20-%20Week%203%20Exercise%20Answer.ipynb#scrollTo=qxju4ItJKO8F
    import csv
    import random
    #wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv -O ./tmp/training_cleaned.csv
    # Note that I cleaned the Stanford dataset to remove LATIN1 encoding to make it easier for Python CSV reader
    # You can do that yourself with:
    # iconv -f LATIN1 -t UTF8 training.1600000.processed.noemoticon.csv -o training_cleaned.csv
    # I then hosted it on my site to make it easier to use in this notebook
    ## Lesson Example Using imdb_reviews/subwords8k

    embedding_dim = 100
    max_length = 16
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    test_portion = .1

    # Your dataset size here. Experiment using smaller values (i.e. 16000), but don't forget to train on at least 160000 to see the best effects
    training_size = 160000

    corpus = []




    num_sentences = 0

    with open("./tmp/training_cleaned.csv" , encoding='UTF8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',' )
        for row in reader:
            # Your Code here. Create list items where the first item is the text, found in row[5],
            # and the second is the label. Note that the label is a '0' or a '4' in the text. When it's the former, make
            # your label to be 0, otherwise 1. Keep a count of the number of sentences in num_sentences
            list_item = []
            # YOUR CODE HERE
            list_item.append(row[5])
            this_label = row[0]
            if this_label == '0': ## 0
                list_item.append(0)
            else: ## 4
                list_item.append(1)
            num_sentences = num_sentences + 1
            corpus.append(list_item)


    print(num_sentences)
    print(len(corpus))
    print(corpus[1])
    # Expected Output:
    # 1600000
    # 1600000
    # ["is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!", 0]

    sentences = []
    labels = []
    random.shuffle(corpus)
    for x in range(training_size):
        sentences.append(corpus[x][0])
        labels.append(corpus[x][1])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    split = int(test_portion * training_size)

    test_sequences = padded[0:split]
    training_sequences = padded[split:training_size]

    test_labels = labels[0:split]
    training_labels = labels[split:training_size]

    print(vocab_size)
    print(word_index['i'])
    # Expected Output
    # 138858
    # 1

    # Note this is the 100 dimension version of GloVe from Stanford
    # I unzipped and hosted it on my site to make this notebook easier
    ### !wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt -O ./tmp/glove.6B.100d.txt
    embeddings_index = {};
    with open('./tmp/glove.6B.100d.txt' , encoding='UTF8') as f:
        for line in f:
            values = line.split();
            word = values[0];
            coefs = np.asarray(values[1:], dtype='float32');
            embeddings_index[word] = coefs;

    embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim));
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word);
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector;
    print(len(embeddings_matrix))
    # Expected Output
    # 138859

    training_padded = np.array(training_sequences)
    training_labels = np.array(training_labels)
    testing_padded = np.array(test_sequences)
    testing_labels = np.array(test_labels)

    ''' Model '''

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length, weights=[embeddings_matrix],
                                  trainable=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    num_epochs = 50
    history = model.fit(training_padded, training_labels, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels), verbose=2)

    print("Training Complete")

    ''' Draw '''
    import matplotlib.image  as mpimg
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])

    plt.figure()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])

    plt.figure()

def LTSM () :
    # Single    : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201a.ipynb#scrollTo=RFEXtKtqNARB
    # Multi     : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201b.ipynb
    # Get the data
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    tokenizer = info.features['text'].encoder
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
        ## Bidirectional : 양방향 ! 따라서 양방향으로 2배가 된다
        # return_sequences : 쌓을 때 사용
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences= True)),   ## 쌓을 때 사용
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),   ## 64: 원하는 출력 값
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    NUM_EPOCHS = 10
    history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

    import matplotlib.pyplot as plt

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
def usingConv () :
    # Get the data
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    tokenizer = info.features['text'].encoder

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        ##단어는 필터 크기로 그룹화 된다
        #컨볼루션은 원하는 크기에 매핑이 가능한다
        # 5단어마다 128 필터가 있따
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    NUM_EPOCHS = 10
    history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

    import matplotlib.pyplot as plt

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')

if __name__ == '__main__' :
    # LTSM()
    # SwitchModel()
    # SwitchModel_sarcasm()
    Exercise_Large_Dataset()