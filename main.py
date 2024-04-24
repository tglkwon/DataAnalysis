import pandas as pd
import pickle as pk
import numpy as np

# visualization
# import matplotlib.pyplot as plt
# import scikitplot as skplt
# import sklearn_evaluation

filename = 'aiml_test_data.xlsx'

def load_dataset(filename):
    dataset = pd.read_excel(filename)       

    print(dataset.shape);    print(dataset.head(5));    print(dataset.columns)

    feature_names = dataset.head(1)
    target = 'paid_amount'

    return feature_names, target, dataset

# execute the function
feature_names, target, dataset = load_dataset(filename) 