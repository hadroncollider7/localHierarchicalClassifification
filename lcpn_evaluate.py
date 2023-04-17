import numpy as np
import pandas as pd
from dataset_query import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from hiclass import LocalClassifierPerNode
from sklearn.pipeline import Pipeline
import time
import pickle
from hiclass.metrics import f1, precision, recall
from sklearn.metrics import classification_report, precision_recall_fscore_support
from numpy_methods import concatStringArray
import os
os.system("cls")

def data_preprocess(data):
    # Remove rows with NaN in Column
    data.dropna(
        subset=["Text", "Cat1", "Cat2", "Cat3"]
    )
    # Rebuild index
    data.reset_index(drop=True, inplace=True)
    X1 = data['Title'].to_numpy(dtype='U')
    X2 = data['Text'].to_numpy(dtype='U')
    X = concatStringArray(X1, X2)
    y = data[['Cat1', 'Cat2', 'Cat3']].to_numpy(dtype='U')
    print('X shape: {0}\ny shape: {1}'.format(np.shape(X), np.shape(y)))
    return data, X, y

# Load model from disk
filepath = './models/lcppn.sav'
pipeline = pickle.load(open(filepath, "rb"))

# Load a dataset
rd_path = ['./data/amazonProductReviews/train_40k.csv', './data/amazonProductReviews/val_10k.csv']
data_train = load_dataset(rd_path[0])
data_test = load_dataset(rd_path[1])
data_val = pd.concat([data_train, data_test], ignore_index=True)
print('data shape: ', data_val.shape)


data_test, X, y = data_preprocess(data_test)

predictions = pipeline.predict(X)
print("predictions shape: ", predictions.shape)
# os.system("cls")
# print('X shape: {0}'.format(np.shape(X)))
print('predictions shape: {0}'.format(np.shape(predictions)))
# print(predictions)

# Compute f-score
f1_hierarchical = f1(y, predictions)
precision_hierarchical = precision(y, predictions)
recall_hierarchical = recall(y, predictions)
print(f"precision: {precision_hierarchical}")
print(f"recall: {recall_hierarchical}")
print(f"f-score: {f1_hierarchical}\n")

print(classification_report(y[:,2], predictions[:,2]))
# print(precision_recall_fscore_support(y[:,2], predictions[:,2]))