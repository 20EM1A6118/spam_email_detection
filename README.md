Spam Email Detection

Spam Email Detection is a simple machine learning project built using Python.
It checks whether a given email or SMS message is spam or not.

Features
Classifies messages as spam or not spam
Uses a real SMS spam dataset
Trained using a machine learning model
Provides accuracy of the model
Allows testing with new messages

Tech Stack
Python
Pandas
NumPy
Scikit-learn

How It Works
The dataset is loaded and cleaned using Pandas.
Text messages are converted into numerical format using CountVectorizer.
The data is split into training and testing sets.
A Naive Bayes model is trained on the training data.
The model predicts whether a new message is spam or not.

Result
The model achieves around 95% accuracy on test data.

How to Run
Install required libraries using pip install pandas numpy scikit-learn.
Run the Python file using python spam_detection.py.

Author
Created by Sushma Rani Kommireddy (B.Tech AI & ML, 2024)
