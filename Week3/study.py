from __future__ import absolute_import, division, print_function, unicode_literals
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
print(tfds.__version__)
import tensorflow as tf
print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus: # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e: # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
        print(e)



def Week1_study () :
    Sarasm = True
    if Sarasm:
    ## Sarasm
        vocab_size = 10000
        embedding_dim = 16
        max_length = 100
        trunc_type = 'post'
        padding_type = 'post'
        oov_tok = "<OOV>"
        training_size = 20000
        with open("./tmp/sarcasm.json" , 'r') as f :
            datastore = json.load(f)

    # print(datastrore[:5])
        sentences = []
        labels = []

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


        # Need this block to get it to work with TensorFlow 2.x
        import numpy as np
        training_padded = np.array(training_padded)
        training_labels = np.array(training_labels)
        testing_padded = np.array(testing_padded)
        testing_labels = np.array(testing_labels)

        '''
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        '''
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        num_epochs = 30
        history = model.fit(training_padded, training_labels, epochs=num_epochs,
                            validation_data=(testing_padded, testing_labels), verbose=2)
    CSV_ = False
    if CSV_ :
        import csv
        # Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
        stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
                     "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could",
                     "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
                     "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
                     "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
                     "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of",
                     "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
                     "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that",
                     "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
                     "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
                     "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what",
                     "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
                     "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
                     "yourself", "yourselves"]
        setences = []
        labels = []

        with open("./tmp/bbc-text.csv" , 'r') as f :
            reader = csv.reader(f ,delimiter =',')
            next(reader)
            for row in reader :
                print(row)
                labels.append(row[0])
                sentence = row[1]
                for n in stopwords :
                    token = " " + n + " "
                    sentence = sentence.replace(token , " ")
                    sentence = sentence.replace("  " , " ")
                setences.append(sentence)
                print(sentence)
def LTSM_study()  :

    # Get the data
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)])
        except RuntimeError as e:
            # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
            print(e)

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
        tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    NUM_EPOCHS = 10
    history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)
if __name__ == '__main__' :
    # Week1_study()
    LTSM_study()