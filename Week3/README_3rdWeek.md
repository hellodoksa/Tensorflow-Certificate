# Tensorflow Certificate Course 3 
Over the last four weeks you've gotten a grounding in how to do Natural Language processing with TensorFlow and Keras. You went from first principles -- basic Tokenization and Padding of text to produce data structures that could be used in a Neural Network. 

You then learned about embeddings, and how words could be mapped to vectors, and words of similar semantics given vectors pointing in a similar direction, giving you a mathematical model for their meaning, which could then be fed into a deep neural network for classification.

From there you started learning about sequence models, and how they help deepen your understanding of sentiment in text by not just looking at words in isolation, but also how their meanings change when they qualify one another. 

You wrapped up by taking everything you learned and using it to build a poetry generator! 

This is just a beginning in using TensorFlow for natural language processing. I hope it was a good start for you, and you feel equipped to go to the next level! 

## Week 1
* Tokenizer
    -  https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%201.ipynb
* Padding 
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%202.ipynb#scrollTo=rX8mhOLljYeM
* Sarcasm example (json)
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%203.ipynb#scrollTo=OkaBMeNDwMel
    
* Exercise 1- Explore the BBC news archive
    - quiz : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Exercise-question.ipynb
    - answer : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Exercise-answer.ipynb
    
## Week 2  -imdb_reviews
* How to use tensorflow_datasets
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%201.ipynb#scrollTo=_IoM4VFxWpMR
   
* Sarcasm - Embedding , GlobalAveragePooling1D
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%202.ipynb
    
* tensorflow_datasets - imdb_reviews/subwords8k
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%203.ipynb
    
* Exercise 2 - BBC news archive
    - quiz : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Question.ipynb
    - answer : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Answer.ipynb
    
## Week 3 - LSTM 
* IMDB Subwords 8K with Single Layer LSTM
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201a.ipynb

* IMDB Subwords 8K with Multi Layer LSTM
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201b.ipynb

* IMDB Subwords 8K with 1D Convolutional Layer
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201c.ipynb

* IMDB Reviews with GRU (and optional LSTM and Conv1D)
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202d.ipynb#scrollTo=nHGYuU4jPYaj

* Exercise 3 - Exploring overfitting in NLP
    - quiz : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP%20Course%20-%20Week%203%20Exercise%20Question.ipynb
    - answer : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP%20Course%20-%20Week%203%20Exercise%20Answer.ipynb

## Week 4 - Sequence model 
* Train and Predict 
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%201%20-%20Notebook.ipynb
    
* Poetry 
    - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%202%20-%20Notebook.ipynb

* generating text using a character-based RNN
    - https://www.tensorflow.org/tutorials/text/text_generation
    
* Exercise 4 -  Using LSTMs, see if you can write Shakespeare!
    - quiz : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP_Week4_Exercise_Shakespeare_Question.ipynb
    - answer : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP_Week4_Exercise_Shakespeare_Answer.ipynb
    