import numpy as np
import pandas as pd
import pickle

from functions import clean_text, print_metrics, plot_confusion_matrix

from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PATH = "data/new-data/"
pkl_filename = "models/logreg_model.pkl"

# initialize label encoder
label_encoder = preprocessing.LabelEncoder()

new = pd.read_csv(PATH + 'new_data.csv', header=0)
new_clean = new[new["statement"].map(len) <= 280].drop_duplicates()  # drop posts longer than 280 characters & drop duplicates
X_new, y_new = new_clean["statement"], new_clean["label"]

X_new = X_new.map(lambda x: clean_text(x))
y_new = label_encoder.fit_transform(y_new)

with open(pkl_filename, 'rb') as file:
    new_pipeline = pickle.load(file)

svm_pred_new = new_pipeline.predict(X_new) # test set 
svm_pred_proba = new_pipeline.predict_proba(X_new) 

print(X_new[:1])
string_ = clean_text("Two years into the pandemic, we ask “what will it take to end its acute phase?” Today, we stress the urgency of equitable access to life-saving #COVID19 tools.")
x_ = pd.Series(np.array([string_]))
print(x_)
print("probas of x_df: ", new_pipeline.predict_proba(x_))

# New data
# display results
print_metrics(svm_pred_new, y_new)
plot_confusion_matrix(confusion_matrix(y_new,svm_pred_new),target_names=['fake','real'], normalize = False, \
                      title = 'Confusion matix of SVM on new data')