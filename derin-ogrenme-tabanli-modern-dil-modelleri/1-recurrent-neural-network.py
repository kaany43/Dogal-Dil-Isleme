'''
Recurrent Neural Network (RNN) for Sentiment Analysis on IMDB Dataset
This script builds and trains an RNN model using Keras to classify movie reviews
'''

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('data/IMDB Dataset.csv')

### Text cleaning and preprocessing
data['review'] = data['review'].str.replace('<br />', ' ')
data['review'] = data['review'].str.lower()
data['review'] = data['review'].str.replace('[^a-zA-Z]', ' ', regex=True)
data['review'] = data['review'].str.split()

# Remove stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
data['review'] = data['review'].apply(lambda x: [word for word in x if word not in stop_words])
data['review'] = data['review'].apply(lambda x: ' '.join(x))


# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['review'])
X = tokenizer.texts_to_sequences(data['review'])

x_seq = pad_sequences(X, maxlen=500)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(data['sentiment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x_seq, y, test_size=0.2, random_state=42)


# Build Word2Vec embeddings
sentences = data['review'].apply(lambda x: x.split())
w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

embedding_dim = 100
VOCAB_SIZE = min(5000, len(tokenizer.word_index)) + 1

embedding_matrix = np.zeros((VOCAB_SIZE, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i >= 5000:
        continue
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]


# Build RNN model
model = Sequential()

model.add(Embedding(input_dim=VOCAB_SIZE,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=500,
                    trainable=False))

model.add(SimpleRNN(128, activation='tanh', dropout=0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")