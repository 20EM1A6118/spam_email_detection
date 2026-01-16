import pandas as pd 
import numpy as np
data=pd.read_csv("spam.csv", encoding='latin-1')
data=data[['v1','v2']]
data.columns=['label' ,'message']
print(data.head())
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['message']
y = data['label']
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
test_email = ["You won a free prize! Click now"]
test_vector = vectorizer.transform(test_email)

result = model.predict(test_vector)

if result[0] == 1:
    print("Spam Email")
else:
    print("Not Spam")




