#Imported libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


# Reads the first 6000 lines of our own data.

df = pd.read_csv('data/ourData/final.csv', nrows=6000)

# Replaces 'type' with binary classification.

i = 0
for x in range(len(['type'])):
    if x == "reliable":
        df.replace(to_replace=df.iloc[i, 1],
                   value='0',
                   inplace=True)
        i = i + 1
    else:
        df.replace(to_replace=df.iloc[i, 1],
                   value='1',
                   inplace=True)
        i = i + 1

# Fill all null values with an empty characters in order to run tfidf_vecotrizer.

df = df.fillna(' ')

# The data needed are stored in variables.

X = df['content'].values
Y = df['type'].values

# Make vectorizer matrix that analyses the term Frequency – Inverse Document.
# It computes the frequency of the words in content.

# parameters in the vectorize:
# stop_words = 'english'      - checks for stop words in english.
# max_df = 0.7                - detects and filter stop words based on intra corpus document frequency of terms.


vectorize = TfidfVectorizer(stop_words='english', max_df=0.7)
vectorize.fit(X)
X = vectorize.transform(X)

# splits the data

# parameters in the train_test_split:
# test_size = 0.2            - proportion of the dataset included in the test split (0.2 = 20%).
# random_state = 10          - shuffles the data before applying the split.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

#resize the distribution of values in X_train

#parameters in the StandardScaler:
# with_mean=False              - if True, centers the data before scaling, but can lead to lack of memory. 

scale = StandardScaler(with_mean=False)
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

#Classifier implementing the k-nearest neighbors vote

# parameters in the StandardScaler:
# n_neighbors=5                - Number of neighbors to use by default for kneighbors queries. 

classify = KNeighborsClassifier(n_neighbors=5)
classify.fit(X_train, Y_train)
y_pred = classify.predict(X_test)

#Print statements in order to see result in terminal. 
print("KNN Results:")
print("Number of mislabeled points out of a total %d points : %d" %
      (X_test.shape[0], (Y_test != y_pred).sum()))
numb_of_correct = X_test.shape[0]-(Y_test != y_pred).sum()
print("Accuracy:",float("{0:.4f}".format(numb_of_correct/X_test.shape[0]*100)), "%")
print("F1 Score Weighted:", f1_score(Y_test, y_pred, average='weighted'))
