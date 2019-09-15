import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

email_data = pd.read_csv("/Users/darling/Downloads/sms_raw_NB.csv",encoding = "ISO-8859-1")

import re
stop_words = []
with open("/Users/darling/Downloads/stop.txt") as f:
    stop_words = f.read()
stop_words = stop_words.split("\n")

def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))
#here we are cleaning the data
email_data.text = email_data.text.apply(cleaning_text)

email_data.shape
email_data = email_data.loc[email_data.text != " ",:]
email_data.shape

def split_into_words(i):
    return [word for word in i.split(" ")]

from sklearn.model_selection import train_test_split
email_train,email_test = train_test_split(email_data,test_size=0.3)

#----------------MODEL BUILDING------------------

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_emails_matrix,email_train.type)
train_pred_m = classifier_mb.predict(train_emails_matrix)
accuracy_train_m = np.mean(train_pred_m==email_train.type) # 98%

test_pred_m = classifier_mb.predict(test_emails_matrix)
accuracy_test_m = np.mean(test_pred_m==email_test.type) # 96%

# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_emails_matrix.toarray(),email_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_emails_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==email_train.type) # 90%

test_pred_g = classifier_gb.predict(test_emails_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==email_test.type) # 83%
