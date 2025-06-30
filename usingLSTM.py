import os

import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import re
import string
from tensorflow.keras.utils import to_categorical
from keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding, LSTM

data = pd.read_csv('fake_or_real_news.csv')
data = data.dropna(subset=['text'])

data = data.head(1000)

vocab_size = 5000

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    text = ' '.join(text.split())
    return text

preprocessed_text = data['text'].apply(preprocess_text).tolist()

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(preprocessed_text)
words_len = len(tokenizer.word_index)

sequences = tokenizer.texts_to_sequences(preprocessed_text)

input_sequences = []

for sequence in sequences:
    if len(sequence) < 21:
        continue
    for i in range(20, len(sequence)):
        input_seq = sequence[i - 20:i + 1]
        input_sequences.append(input_seq)

input_sequences = np.array(input_sequences)[:50000]

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)

Y_train = to_categorical(Y_train, num_classes=words_len-1)
Y_test = to_categorical(Y_test, num_classes=words_len-1)

model = Sequential()
model.add(Embedding(words_len-1,100,input_length=20))
model.add(LSTM(150))
model.add(Dense(words_len-1,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train,Y_train,epochs=5,validation_data=(X_test,Y_test),verbose=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len=20):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-max_sequence_len:]  # Keep only last max_sequence_len tokens
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

        predicted = model.predict(token_list, verbose=0)
        predicted_id = np.argmax(predicted, axis=-1)[0]

        # Find the word corresponding to the predicted ID
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                output_word = word
                break

        if output_word == "":
            break

        seed_text += " " + output_word

    return seed_text


print("\nGenerating sample text:")
sample_text = generate_text("the news report", 10, model, tokenizer)
print(sample_text)