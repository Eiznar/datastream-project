import numpy as np
import pandas as pd

from functions import clean_text, print_metrics, plot_confusion_matrix

from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load data
# import nltk
# nltk.download('omw-1.4')

PATH = "data/original-data/"

train = pd.read_csv(PATH + 'Constraint_Train.csv', header=0)
train_clean = train[train["tweet"].map(len) <= 280].drop_duplicates() # drop posts longer than 280 characters & drop duplicates
X_train, y_train = train_clean["tweet"], train_clean["label"]

val = pd.read_csv(PATH + 'Constraint_Val.csv', header=0)
val_clean = val[val["tweet"].map(len) <= 280].drop_duplicates()  # drop posts longer than 280 characters & drop duplicates
X_val, y_val = val_clean["tweet"], val_clean["label"]


test = pd.read_csv(PATH + 'Constraint_Test.csv', header=0)
test_clean = test[test["tweet"].map(len) <= 280].drop_duplicates()  # drop posts longer than 280 characters & drop duplicates
X_test, y_test = test_clean["tweet"], test_clean["label"]

# Preprocess data

# apply clean_text() function to all tweets 
X_train = X_train.map(lambda x: clean_text(x))
X_val = X_val.map(lambda x: clean_text(x))
X_test = X_test.map(lambda x: clean_text(x))

# initialize label encoder
label_encoder = preprocessing.LabelEncoder()

# encode 'fake' as 0 and 'real' as 1 to make target variables machine-readable
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.fit_transform(y_val)
y_test = label_encoder.fit_transform(y_test)


####################### The model ##############################

# create SVM object
svm_clf = SVC(kernel='linear',probability=True, C=10, class_weight='balanced')

# create pipeline
svm_pipeline = Pipeline([
        ('bow', CountVectorizer(ngram_range=(1, 2))), # count term frequency
        ('tfidf', TfidfTransformer()), # downweight words which appear frequently
        ('c', svm_clf) # classifier
])

# train model
fit = svm_pipeline.fit(X_train,y_train)

# make predictions
svm_pred_val = svm_pipeline.predict(X_val) # validation set 
svm_pred_test = svm_pipeline.predict(X_test) # test set 


####################### Evaluation ##############################

# validation set
# display results

print_metrics(svm_pred_val,y_val)
plot_confusion_matrix(confusion_matrix(y_val,svm_pred_val),target_names=['fake','real'], normalize = False, \
                      title = 'Confusion matix of SVM on val data')

# test set
# display results
print_metrics(svm_pred_test,y_test)
plot_confusion_matrix(confusion_matrix(y_test,svm_pred_test),target_names=['fake','real'], normalize = False, \
                      title = 'Confusion matix of SVM on test data')


####################### Testing on unknown data ##############################
PATH = "data/new-data/"

new = pd.read_csv(PATH + 'new_data.csv', header=0)
new_clean = new[new["statement"].map(len) <= 280].drop_duplicates()  # drop posts longer than 280 characters & drop duplicates
X_new, y_new = new_clean["statement"], new_clean["label"]

X_new = X_new.map(lambda x: clean_text(x))
y_new = label_encoder.fit_transform(y_new)


svm_pred_new = svm_pipeline.predict(X_new) # test set 

# New data
# display results
print_metrics(svm_pred_new, y_new)
plot_confusion_matrix(confusion_matrix(y_new,svm_pred_new),target_names=['fake','real'], normalize = False, \
                      title = 'Confusion matix of SVM on new data')