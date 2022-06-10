import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import  accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
nltk.download('stopwords')
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#We've taken inspiration from this article https://machinelearningprojects.net/fake-news-classifier-using-lstm/
# the code takes about 9 min to run

# Reads the first 6000 lines of our own data.

df = pd.read_csv('data/ourData/final.csv', chunksize=1000, nrows=6000)
df = pd.concat(df)
df = df.fillna(' ')
e = 0 
for x in df['type']:
    if x == "reliable":
        df.replace(to_replace=df.iloc[e, 1],
                   value=0,
                   inplace=True)
        e = e + 1 
    else:
        df.replace(to_replace=df.iloc[e, 1],
            value=1,
            inplace=True) 
        e = e + 1

X = df['content']
y = df['type']

ps = PorterStemmer()
corpus = []
for i in range(len(X)):
    text = X[i]
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = text.split()
    text = [ps.stem(t) for t in text if t not in stopwords.words('english')]
    corpus.append(' '.join(text))


vocab_size = 5000
sent_len = 20
one_hot_encoded = [one_hot(x, vocab_size) for x in corpus]
one_hot_encoded = pad_sequences(one_hot_encoded, maxlen=sent_len)

X = np.array(one_hot_encoded)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


no_of_output_features = 40
model = Sequential()
model.add(Embedding(vocab_size, no_of_output_features, input_length=sent_len))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=5)

pred = (model.predict(X_test) > 0.5).astype("int32")

score = accuracy_score(y_test, pred)

print("Neurel Network:")
print(f'Accuracy: {round(score*100,2)}%')
print("Number of mislabeled points out of a total %d points : %d" %
      (X_test.shape[0], (y_test != pred).sum()))
print("F1 Score Weighted:", f1_score(y_test, pred, average='weighted'))

