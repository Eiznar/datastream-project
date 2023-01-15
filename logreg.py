import numpy as np
import pandas as pd
import pickle

from utils import clean_text, print_metrics, plot_confusion_matrix

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

# Load data
# import nltk
# nltk.download('omw-1.4')

PATH = "data/original-data/"

train = pd.read_csv(PATH + 'Constraint_Train.csv', header=0)
# drop posts that are too long and the duplicates
train = train[train["tweet"].map(len) <= 280].drop_duplicates()
X_train, y_train = train["tweet"], train["label"]

val = pd.read_csv(PATH + 'Constraint_Val.csv', header=0)
# drop posts that are too long and the duplicates
val = val[val["tweet"].map(len) <= 280].drop_duplicates()
X_val, y_val = val["tweet"], val["label"]


test = pd.read_csv(PATH + 'Constraint_Test.csv', header=0)
# drop posts that are too long and the duplicates
test = test[test["tweet"].map(len) <= 280].drop_duplicates()
X_test, y_test = test["tweet"], test["label"]

# Preprocessing

# we clean all the tweets
X_train = X_train.map(clean_text)
X_val = X_val.map(clean_text)
X_test = X_test.map(clean_text)

label_encoder = preprocessing.LabelEncoder()

#  'fake' -> 0, 'real' -> 1
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.fit_transform(y_val)
y_test = label_encoder.fit_transform(y_test)


####################### The model ##############################

# Logistic Regression model
clf = LogisticRegression(max_iter=1000, class_weight='balanced')

# Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(ngram_range=(1, 2))),  # count term frequency
    ('tfidf', TfidfTransformer()),  # downweight words which appear frequently
    ('c', clf)  # classifier
])

# Train
fit = pipeline.fit(X_train, y_train)

# Prediction on val (old data) and test (old data)
pred_val = pipeline.predict(X_val)
pred_test = pipeline.predict(X_test)


####################### Evaluation ##############################

# validation set

print_metrics(pred_val, y_val)
plot_confusion_matrix(confusion_matrix(y_val, pred_val), target_names=['fake', 'real'], normalize=False,
                      title='Confusion matix of Logistic Regression on val data')

# test set

print_metrics(pred_test, y_test)
plot_confusion_matrix(confusion_matrix(y_test, pred_test), target_names=['fake', 'real'], normalize=False,
                      title='Confusion matix of Logistic Regression on test data')


####################### Testing on unknown data ##############################
PATH = "data/new-data/"

new = pd.read_csv(PATH + 'new_data.csv', header=0)
# drop posts longer than 280 characters & drop duplicates
new = new[new["statement"].map(len) <= 280].drop_duplicates()
X_new, y_new = new["statement"], new["label"]

X_new = X_new.map(clean_text)
y_new = label_encoder.fit_transform(y_new)

print(X_new.head())

pred_new = pipeline.predict(X_new)
pred_proba = pipeline.predict_proba(X_new)

# New data
# display results
print_metrics(pred_new, y_new)
plot_confusion_matrix(confusion_matrix(y_new, pred_new), target_names=['fake', 'real'], normalize=False,
                      title='Confusion matix of Logistic Regression on new data')

# Save model

pkl_filename = "models/logreg_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pipeline, file)
