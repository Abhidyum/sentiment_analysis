# importing important libraries

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"D:\kaggle\nlp\train.csv")

df_cleaned = dataset.dropna()
df_cleaned = df_cleaned.reset_index(drop=True)

def remove_urls(text):   # function to remove URLs from the text data
    # Define the regex pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    # Remove URLs from the text using the pattern
    text_without_urls = re.sub(url_pattern, '', text)
    return text_without_urls


for i in range(len(df_cleaned)):
    df_cleaned['text'][i] = remove_urls(df_cleaned["text"][i])        # removing URLs

print(df_cleaned)

# Perform lemmatization on the dataset
lemmatized_dataset = []
lemmatizer = WordNetLemmatizer()

for text in df_cleaned['text']:
    tokens = nltk.word_tokenize(text)  # Tokenize the text into words
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize each word
    lemmatized_text = " ".join(lemmas)  # Join the lemmatized words back into a text
    lemmatized_dataset.append(lemmatized_text)

print(lemmatized_dataset)

y = df_cleaned['sentiment']
y = pd.get_dummies(y, prefix='emotion').values     # one_hot_Encoding traget values

tokenizer = Tokenizer(num_words = 10000, lower= True, oov_token="<OOV>")
tokenizer.fit_on_texts(lemmatized_dataset)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(lemmatized_dataset)

print(sequences)

max_length = 0                            # to find the maximum length of a sentence
for sublist in lemmatized_dataset:
    sublist_length = len(sublist)
    if sublist_length > max_length:
        max_length = sublist_length

print("Length of the longest sublist:", max_length)

train = pad_sequences(sequences, maxlen = 75)
print(train)

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.25, random_state=42)

num_classes = 3

# creating my LSTM model
model = Sequential()
model.add(Embedding(input_dim=10001, output_dim=100, input_length=maxlen))
model.add(LSTM(units=64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
model.add(LSTM(units=32, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(units=num_classes, activation='linear'))

learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=optimizer, metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                   epochs = 5,
                   validation_data = (X_test, y_test),
                   shuffle = True)


# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# checking performance of my model on test data

logits = model.predict(X_test)
predictions = tf.nn.softmax(logits)













