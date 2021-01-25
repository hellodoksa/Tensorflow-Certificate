import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional ,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

def lesson4_1() :
    # https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%201%20-%20Notebook.ipynb#scrollTo=PRnDnCW-Z7qv
    tokenizer = Tokenizer()
    data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

    corpus = data.lower().split("\n")

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    print(tokenizer.word_index)
    print(total_words) ## 단어 총 수

    input_sequences = []
    ## 한 문장당 단어씩 잘라서
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    xs, labels = input_sequences[:, :-1], input_sequences[:, -1]

    # one hot encoding => 263개 중에 맞는거에 1
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    print(tokenizer.word_index['in'])
    print(tokenizer.word_index['the'])
    print(tokenizer.word_index['town'])
    print(tokenizer.word_index['of'])
    print(tokenizer.word_index['athy'])
    print(tokenizer.word_index['one'])
    print(tokenizer.word_index['jeremy'])
    print(tokenizer.word_index['lanigan']) ## Label

    print(xs[6])
    # [0  0  0  4  2 66  8 67 68 69]
    print(ys[6])
    '''
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    '''

    print(xs[5])
    print(ys[5])
    '''
    [ 0  0  0  0  4  2 66  8 67 68]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    '''

    print(tokenizer.word_index)

    model = Sequential()
    model.add(Embedding(total_words, 64, input_length=max_sequence_len - 1))     ## max_sequence_len - 1 => 라벨을 얻기 위해 뒤에서 하나 뺏기 때문

    model.add(Bidirectional(LSTM(20))) ## 양방향을 하면 보다 더 자연스러운 추측이 가능하다 .
    model.add(Dense(total_words, activation='softmax'))     ## total_words => 원핫 인코딩과 같은 크기 , 단어 하나당 하나의 뉴런이 존재

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) ## 범주 예측

    history = model.fit(xs, ys, epochs=500, verbose=1)

    import matplotlib.pyplot as plt

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.show()

    plot_graphs(history, 'accuracy')

    ## 예측
    ''' 
    Laurence 는 없는 단어이기 때문에 패스 하고 있는 단어들을 이용하여 찾는다 
    '''
    seed_text = "Laurence went to dublin"
    next_words = 100

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)

def Poetry() :
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%202%20-%20Notebook.ipynb
# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt -O ./tmp/irish-lyrics-eof.txt
    tokenizer = Tokenizer()

    data = open('./tmp/irish-lyrics-eof.txt').read()

    corpus = data.lower().split("\n")

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    print(tokenizer.word_index)
    print(total_words)

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        print("len token_list : " , len(token_list))
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    print(tokenizer.word_index['in'])
    print(tokenizer.word_index['the'])
    print(tokenizer.word_index['town'])
    print(tokenizer.word_index['of'])
    print(tokenizer.word_index['athy'])
    print(tokenizer.word_index['one'])
    print(tokenizer.word_index['jeremy'])
    print(tokenizer.word_index['lanigan'])

    print(xs[6])
    print(ys[6])

    print(tokenizer.word_index)


    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(total_words, activation='softmax'))
    adam = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.summary()
    history = model.fit(xs, ys, epochs=100, verbose=1)

    print(model)

    import matplotlib.pyplot as plt

    def plot_graphs(history, string):
      plt.plot(history.history[string])
      plt.xlabel("Epochs")
      plt.ylabel(string)
      plt.show()


    plot_graphs(history, 'accuracy')

    seed_text = "I've got a bad feeling about this"
    next_words = 100

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)

def HaveToTest () :
    #https://www.tensorflow.org/tutorials/text/text_generation
    print("https://www.tensorflow.org/tutorials/text/text_generation")

'''
 In this course you’ve done a lot of NLP and text processing. 
 This week you trained with a dataset of Irish songs to create traditional-sounding poetry. 
 For this week’s exercise, you’ll take a corpus of Shakespeare sonnets, and use them to train a model. 
 Then, see if that model can create poetry!
'''
def Exercise() :
    #https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP_Week4_Exercise_Shakespeare_Question.ipynb
    #https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP_Week4_Exercise_Shakespeare_Answer.ipynb
    ## !wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt -O ./tmp/sonnets.txt
    import tensorflow.keras.utils as ku
    tokenizer = Tokenizer()
    data = open('./tmp/sonnets.txt').read()

    corpus = data.lower().split("\n")

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

    label = ku.to_categorical(label, num_classes=total_words)

    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words / 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    ''' 
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 10, 100)           321100    
    _________________________________________________________________
    bidirectional (Bidirectional (None, 10, 300)           301200    
    _________________________________________________________________
    dropout (Dropout)            (None, 10, 300)           0         
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 100)               160400    
    _________________________________________________________________
    dense (Dense)                (None, 1605)              162105    
    _________________________________________________________________
    dense_1 (Dense)              (None, 3211)              5156866   
    =================================================================
    Total params: 6,101,671
    Trainable params: 6,101,671
    Non-trainable params: 0
    _________________________________________________________________
    None
    '''
    history = model.fit(predictors, label, epochs=100, verbose=1)

    import matplotlib.pyplot as plt
    acc = history.history['accuracy']
    loss = history.history['loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()

    plt.show()

    seed_text = "Help me Obi Wan Kenobi, you're my only hope"
    next_words = 100

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)
if __name__ == '__main__' :
    HaveToTest()

    # Poetry()
    Exercise()