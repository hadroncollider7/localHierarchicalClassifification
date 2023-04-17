import pandas as pd
import numpy as np
import os

def df_to_csv(data, rows, filename):
    """Save a small sample of data to view in spreadsheet.

    Args:
        data (pandas dataframe): 
        rows (array): Array with two elements containing the beginning and ending index of the dataframe to be saved.
        filename (string): The filepath
    """
    df = data.loc[rows[0]:rows[1]]
    df.to_csv('{0:s}'.format(filename))
    
def load_dataset(rd_path):
    """Import a csv dataset and returns a pandas dataframe

    Args:
        rd_path (string): _description_

    Returns:
        dataframe: _description_
    """
    data = pd.read_csv(
        rd_path,
        sep=',',
        header=0,
        low_memory=False,
    )
    return data
        


def saveSampleDataset(rd_path, wr_path, rows):
    data = pd.read_csv(
        rd_path,
        sep=',',
        header=0,
        low_memory=False,
    )
    df_to_csv(data, rows, wr_path)
    
if __name__ == "__main__":
    os.system("cls")
    rd_path = './data/amazonProductReviews/train_40k.csv'
    # wr_path = './data/amazonProductReviews/val_10k_sample.csv'
    # saveSampleDataset(rd_path, wr_path, [0,1000])
    
    
    data = pd.read_csv(
        rd_path,
        sep=',',
        header=0,
        low_memory=False,
    )
    print('dataframe shape: {}'.format(data.shape))
    print(data.loc[0]['Cat1'], data.loc[0]['Cat2'])
    
    # View the distribution of specified classes
    print('Cat1 shape: ', data['Cat1'].value_counts().shape)
    print('Cat2 shape: ', data['Cat2'].value_counts().shape)
    print('Cat3 shape: ', data['Cat3'].value_counts().shape)
    # data['Cat1'].value_counts().to_csv('cat1_valueCounts')
    # print('type: ', type(cat_valuesCounts))
    # print(data['Cat2'].value_counts())
    # print(data['Cat3'].value_counts())
    
    
    X_train = load_dataset(rd_path)
    print('dataset shape: ', X_train.shape)