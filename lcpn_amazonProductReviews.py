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
from memory_profiler import profile
import os
os.system("cls")

@profile
def train_hierarchical_classifier(pipeline, X_train, y_train):
    beginning = time.time()
    pipeline.fit(X_train, y_train)
    end = time.time()
    return pipeline, beginning, end


if __name__ == "__main__":
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
    lcpn = LocalClassifierPerLevel(
        local_classifier=base_classifier,
        verbose=0,
        n_jobs=1
    )
    pipeline = Pipeline([
        ('count', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('model', lcpn)
    ])

    pipeline, beginning, end = train_hierarchical_classifier(pipeline, X_train, y_train)
    # Print dataset size
    print(f"Rows: {len(X_train)}")
    print(f"Time elapsed: {(end - beginning)/60} minutes")

    # Save model to disk
    filepath = './models/lcpl.sav'
    pickle.dump(pipeline, open(filepath, 'wb'))

    # Make predictions
    predictions = pipeline.predict(X_test)
    print("predictions shape: ", predictions.shape)

    # Compute f-score
    f1_hierarchical = f1(y_test, predictions)
    precision_hierarchical = precision(y_test, predictions)
    recall_hierarchical = recall(y_test, predictions)
    print(f"f-score: {f1_hierarchical}")
    print(f"precision: {precision_hierarchical}")
    print(f"recall: {recall_hierarchical}")
    