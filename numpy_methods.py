import numpy as np
import os
os.system("cls")

def concatStringArray(X1, X2):
    spacing = np.full(shape=np.shape(X1), fill_value='. ')
    X = np.char.add(X1, spacing)
    X = np.char.add(X, X2)
    return X



if __name__ == "__main__":
    x = np.array([])
    print(x)
    x = np.append(x,1)
    print(x)
    x = np.append(x,2)
    print(x)