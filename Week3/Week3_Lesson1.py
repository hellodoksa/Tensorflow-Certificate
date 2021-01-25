import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
def Example1 () :
    setence = [
        'I love my dog',
        'I love my cat',
        'You love my cat',
        'Do you think my dog is amazing?',
    ]

    '''" num_words=100" 
    이 하이퍼파라미터를 설정함으로써, 토큰화자가 할 일은 
    볼륨별로 상위 100개의 단어를 가져와서 인코딩하는 것 
    학습 시간이 길다. 
    '''
    tokenizer = Tokenizer(num_words= 100 ,oov_token="<OOV>") # 모르는 단어는 OOV로 대체한다.
    tokenizer.fit_on_texts(setence) ## 데이터를 가져와서 인코딩
    word_index = tokenizer.word_index ## 키(=Word) 값이 포함된 dict(=Token)을 봔한하는 단어 색인을 제공
    print(word_index)

    sequences = tokenizer.texts_to_sequences(setence)
    print(sequences)

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded = pad_sequences(sequences ,
                           padding="pre" , # pre or post
                           maxlen=5 , truncating="post") # 최대 길이보다 크면 앞부터 잘려라!
    print(padded)

# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%202.ipynb#scrollTo=BZSlp3DAjdYf
def colab_original() :
    sentences = [
        'I love my dog',
        'I love my cat',
        'You love my dog!',
        'Do you think my dog is amazing?'
    ]

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>") # 새로운 토큰
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(sentences)

    padded = pad_sequences(sequences, maxlen=5)
    print("\nWord Index = ", word_index)
    '''     Word Index =  {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}     '''

    print("\nSequences = ", sequences)
    '''    Sequences =  [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]    '''

    print("\nPadded Sequences:")
    print(padded)

    # Try with words that the tokenizer wasn't fit to
    test_data = [
        'i really love my dog',
        'my dog loves my manatee'
    ]

    test_seq = tokenizer.texts_to_sequences(test_data)
    print("\nTest Sequence = ", test_seq)

    padded = pad_sequences(test_seq, maxlen=10)
    print("\nPadded Test Sequence: ")
    print(padded)

def KaggleExample () :
#wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json -O ./tmp/sarcasm.json
    with open("./tmp/sarcasm.json", 'r') as f:
        datastore = json.load(f)

    sentences = []
    labels = []
    urls = []
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])


    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index
    print(len(word_index))
    print(word_index)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding='post')
    print(padded[0])
    print(padded.shape) # (26709, 40)

def BBC_test () :
    # https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Exercise-answer.ipynb
    # #wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv -O ./tmp/bbc-text.csv
    import csv
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
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
    sentences = []
    labels = []
    with open("./tmp/bbc-text.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            sentence = row[1]
            for word in stopwords:
                token = " " + word + " "
                sentence = sentence.replace(token, " ")
                sentence = sentence.replace("  ", " ")
            sentences.append(sentence)

    print(len(sentences))
    print(sentences[0])

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print(len(word_index))

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding='post')
    print(padded[0])
    print(padded.shape)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_word_index = label_tokenizer.word_index
    label_seq = label_tokenizer.texts_to_sequences(labels)
    print(label_seq)
    print(label_word_index)

if __name__ == '__main__' :
    # Example1()
    # colab_original()
    # KaggleExample()
    BBC_test()