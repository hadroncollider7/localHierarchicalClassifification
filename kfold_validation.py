import numpy as np
import pandas as pd
from dataset_query import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
from sklearn.pipeline import Pipeline
import time
import pickle
from hiclass.metrics import f1, precision, recall
from numpy_methods import concatStringArray
from sklearn.model_selection import KFold
import os
os.system("cls")


# Load the training set
rd_path = './data/amazonProductReviews/train_40k.csv'
data_train = load_dataset(rd_path)
# data_test = load_dataset(rd_path[1])
# data = pd.concat([data_train, data_test], ignore_index=True)
data = data_train
print('data shape: ', data.shape)

# Remove rows with NaN in Column
data.dropna(
    subset=["Title", "Text", "Cat1", "Cat2", "Cat3"], inplace=True
)
# Rebuild index
data.reset_index(drop=True, inplace=True)

# Concatenate the Title and Text features of each row
X1 = data['Title'].to_numpy(dtype='U')
X2 = data['Text'].to_numpy(dtype='U')
print('X1 shape: {0}\nX2 shape: {1}'.format(np.shape(X1), np.shape(X2)))
print('X1 type: {0}\nX2 type: {1}'.format(type(X1[0]), type(X2[0])))
X = concatStringArray(X1, X2)
print(f"X shape: {np.shape(X)}")
# print(X[:5])


y = data[['Cat1', 'Cat2', 'Cat3']].to_numpy(dtype='U')
print('X shape: {0}\ny shape: {1}'.format(np.shape(X), np.shape(y)))

# Split training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# os.system("cls")
print('X_train shape: {0}\ny_train shape: {1}'.format(np.shape(X_train), np.shape(y_train)))
print('X_test shape: {0}\ny_test shape: {1}'.format(np.shape(X_test), np.shape(y_test)))

# Build ML pipeline
base_classifier = LogisticRegression(
    max_iter=10000,
    n_jobs=1
)
# base_classifier = RandomForestClassifier()
lcpn = LocalClassifierPerNode(
    local_classifier=base_classifier,
    verbose=0,
    n_jobs=1
)
pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', lcpn)
])

hP_score = []
hR_score = []
hF_score = []

# Kfold cross-validation
k = 5
kf = KFold(n_splits=k, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X_train):
    start_time = time.time()
    X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
    y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]
    
    pipeline.fit(X_train_kf, y_train_kf)
    predictions = pipeline.predict(X_test_kf)
    
    hP_score.append(precision(y_test_kf, predictions))
    hR_score.append(recall(y_test_kf, predictions))
    hF_score.append(f1(y_test_kf, predictions))
    end_time = time.time()
    print(f"Time elapsed: {(end_time - start_time)/60} minutes")

print("hP: ", hP_score)
print("hR: ", hR_score)
print("hF: ", hF_score)
print("mean hP: ", sum(hP_score)/k)
print("mean hR: ", sum(hR_score)/k)
print("mean hF: ", sum(hF_score)/k)