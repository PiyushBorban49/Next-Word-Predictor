from random import shuffle

from pygments.lexers.asn1 import word_sequences
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from Tools.i18n.makelocalealias import optimize
from click.core import batch
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import  LSTM,Dense,Activation
import random
import pickle
import pandas as pd
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics

text_df = pd.read_csv("fake_or_real_news.csv")
text_df.dropna(subset=["id","title","text","label"])
# print(text_df)

text = list(text_df.text.values)
joint_text = "".join(text)
# print(joint_text)

partial_text = joint_text[:250000]

tokenizer = RegexpTokenizer(f"\w+")
tokens = tokenizer.tokenize(partial_text.lower())

# print(tokens)
unique_tokens = np.unique(tokens)
unique_token_index = {token : idx for idx,token in enumerate(unique_tokens)}
# print(unique_token_index)

n_words = 10
input_words = []
next_words = []
for i in range(len(tokens)-n_words):
    input_words.append(tokens[i:i+n_words])
    next_words.append(tokens[i+n_words])

X = np.zeros((len(input_words),n_words,len(unique_tokens)),dtype=bool)
Y = np.zeros((len(next_words),len(unique_tokens)),dtype=bool)
for i,words in enumerate(input_words):
    for j,word in enumerate(words):
        X[i, j, unique_token_index[word]] = 1
    Y[i, unique_token_index[next_words[i]]] = 1


# model = Sequential()
# model.add(LSTM(128,input_shape=(n_words,len(unique_tokens)),return_sequences = True))
# model.add(LSTM(128))
# model.add(Dense(len(unique_tokens)))
# model.add(Activation('softmax'))
#
# model.compile(
#     loss="categorical_crossentropy",
#     optimizer=RMSprop(learning_rate=0.01),
#     metrics=['accuracy']
# )
# model.fit(X,Y,batch_size=128,epochs=30,shuffle=True)

# model.save("NextWord.keras")
model = load_model("NextWord.keras")

def predict_next_word(input_text,n_best):
    input_text = input_text.lower()
    X = np.zeros((1,n_words,len(unique_tokens)))
    for i,word in enumerate(input_text.split()):
        X[0,i,unique_token_index[word]] = 1

    predictions = model.predict(X)[0]
    return np.argpartition(predictions,-n_best)[-n_best:]

possible = predict_next_word("I will have to look into this because I",5)

# print(possible)
#
# for i in possible:
#     print(unique_tokens[i])

def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    words = tokenizer.tokenize(input_text)
    X = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(words[-n_words:]):
        if word in unique_token_index:
            X[0, i, unique_token_index[word]] = 1

    predictions = model.predict(X, verbose=0)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]


def generate_text(input_text, text_length, creativity=5):
    word_sequence = tokenizer.tokenize(input_text.lower())

    for _ in range(text_length):
        if len(word_sequence) < n_words:
            sub_sequence = [""] * (n_words - len(word_sequence)) + word_sequence
        else:
            sub_sequence = word_sequence[-n_words:]

        try:
            pred_indices = predict_next_word(" ".join(sub_sequence), creativity)
            choice = unique_tokens[np.random.choice(pred_indices)]
        except Exception as e:
            print(f"[Warning] Using fallback due to: {e}")
            choice = np.random.choice(unique_tokens)

        word_sequence.append(choice)

    return " ".join(word_sequence)


print(generate_text("I will have to look into this because I",100,5))