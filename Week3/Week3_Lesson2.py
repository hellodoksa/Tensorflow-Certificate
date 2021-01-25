import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


print(tf.__version__)

def test1() :
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%201.ipynb
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

    train_data, test_data = imdb['train'], imdb['test']

    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    # str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    oov_tok = "<OOV>"


    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

    ## Decoded
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    print(decode_review(padded[3]))
    print(training_sentences[3])


    model = tf.keras.Sequential([
        ##The top layer of this is going to be an embedding, the embedding is going to be my vocab size,
        # the embedding dimensions that I wanted to use, I had specified 16.
        # My input length for that is 120, which is the maximum length of the reviews.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.GlobalAveragePooling1D() ,
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()


    num_epochs = 10
    model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

    e = model.layers[0]
    weights = e.get_weights()[0]
    ## 단어당 16 벡터를 가지고 있다.
    print(weights.shape) # shape: (vocab_size, embedding_dim)


    import io

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for word_num in range(1, vocab_size):
      word = reverse_word_index[word_num]
      embeddings = weights[word_num]
      out_m.write(word + "\n")
      out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()

    sentence = "I really think this is amazing. honest."
    sequence = tokenizer.texts_to_sequences([sentence])
    print(sequence)

def test2 () :
    # wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json -O ./tmp/sarcasm.json
    #https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%202.ipynb
    import json
    import tensorflow as tf

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    vocab_size = 10000
    embedding_dim = 16
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    with open("./tmp/sarcasm.json", 'r') as f:
        datastore = json.load(f)

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

    display = True
    if display :
        import matplotlib.pyplot as plt

        def plot_graphs(history, string):
            plt.plot(history.history[string])
            plt.plot(history.history['val_' + string])
            plt.xlabel("Epochs")
            plt.ylabel(string)
            plt.legend([string, 'val_' + string])
            plt.show()

        plot_graphs(history, "accuracy")
        plot_graphs(history, "loss")

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_sentence(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    print(decode_sentence(training_padded[0]))
    print(training_sentences[2])
    print(labels[2])

    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape)  # shape: (vocab_size, embedding_dim)

def test_sub() :
    # 하위 단위로 단어를 자른다. => 의미 없음
    # https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%203.ipynb

    import tensorflow as tf
    print(tf.__version__)

    import tensorflow_datasets as tfds
    imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

    train_data, test_data = imdb['train'], imdb['test']
    tokenizer = info.features['text'].encoder
    print(tokenizer.subwords)

    sample_string = 'TensorFlow, from basics to mastery'

    tokenized_string = tokenizer.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer.decode(tokenized_string)
    print('The original string: {}'.format(original_string))

    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer.decode([ts])))

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = train_data.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
    test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))
    embedding_dim = 64
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    num_epochs = 10
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

    import matplotlib.pyplot as plt

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

def BBC_test2() :
    #https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Question.ipynb
    #https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Answer.ipynb
    import csv
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences


    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_portion = .8

    sentences = []
    labels = []
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                 "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                 "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                 "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]
    print(len(stopwords))

    with open("./tmp/bbc-text.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) ## 한 줄 빼기
        for row in reader:
            labels.append(row[0])
            sentence = row[1]
            for word in stopwords:
                token = " " + word + " "
                sentence = sentence.replace(token, " ")
            sentences.append(sentence)

    print(len(labels))
    print(len(sentences))
    print(sentences[0])

    ## Train
    train_size = int(len(sentences) * training_portion)
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]

    ## Validation
    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]

    print(train_size)
    print(len(train_sentences))
    print(len(train_labels))
    print(len(validation_sentences))
    print(len(validation_labels))

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

    print(len(train_sequences[0]))
    print(len(train_padded[0]))

    print(len(train_sequences[1]))
    print(len(train_padded[1]))

    print(len(train_sequences[10]))
    print(len(train_padded[10]))


    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

    print(len(validation_sequences))
    print(validation_padded.shape)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    print('training_label_seq[0] : ' ,training_label_seq[0])
    print('training_label_seq[1] : ' ,training_label_seq[1])
    print('training_label_seq[2] : ' ,training_label_seq[2])
    print(training_label_seq.shape)

    print('validation_label_seq[0] : ' , validation_label_seq[0])
    print('validation_label_seq[1] : ' , validation_label_seq[1])
    print('validation_label_seq[2] : ' , validation_label_seq[2])
    print(validation_label_seq.shape)

    model = tf.keras.Sequential([
        ## vocab_size : 내 어휘 크기
        ## embedding_dim : 출력 임베딩
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    num_epochs = 30
    history = model.fit(train_padded, training_label_seq, epochs=num_epochs,
                        validation_data=(validation_padded, validation_label_seq), verbose=2)

    import matplotlib.pyplot as plt

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    download=False
    if download :
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

        def decode_sentence(text):
            return ' '.join([reverse_word_index.get(i, '?') for i in text])

        e = model.layers[0]
        weights = e.get_weights()[0]
        print(weights.shape)  # shape: (vocab_size, embedding_dim)

        import io

        out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
        out_m = io.open('meta.tsv', 'w', encoding='utf-8')
        for word_num in range(1, vocab_size):
            word = reverse_word_index[word_num]
            embeddings = weights[word_num]
            out_m.write(word + "\n")
            out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
        out_v.close()
        out_m.close()
if __name__ == '__main__' :
    # test1()
    # test2()
    # test_sub()
    BBC_test2()