import pandas as pd
import pickle as pk
import numpy as np
import os

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# import scikitplot as skplt
# import sklearn_evaluation
 
base_dir = '~/project/ExploratoryDataAnalysis'
excel_file = 'aiml_test_data.xlsx'
filename = os.path.join(base_dir, excel_file)

def load_dataset(filename):
    dataset = pd.read_excel(filename, sheet_name='Sheet1', header=0, na_values='NaN')       

    print(dataset.shape);    print(dataset.head(5));    print(dataset.isnull().sum())

    feature_names = dataset.head(1)
    target = 'paid_amount'

    return feature_names, target, dataset

# execute the function
feature_names, target, dataset = load_dataset(filename) 

## Data preprocessing
def preprocessing(dataset):
    cols1 = ['port_of_loading','port_of_discharge']
    cols2 = ['HSCODE','is_coc','paid_amount']
  
    pol, port_of_loading = pd.factorize(dataset['port_of_loading'])
    print(pol); print(port_of_loading)

## Data Summarisation (Descriptive Statistics)
def summariseDataset(dataset):
    cols1 = ['port_of_loading','port_of_discharge']
    cols2 = ['HSCODE','is_coc','paid_amount']
    cols3 = ['cargo_weight','teu','paid_amount']    
    # shape
    print(dataset[cols1].shape)
    print(dataset[cols2].shape)
    print(dataset[cols3].shape)    
    # head
    print(dataset[cols1].head(5))
    print(dataset[cols2].head(5))
    print(dataset[cols3].head(5))    
    # descriptions
    print(dataset[cols1].describe())
    print(dataset[cols2].describe())    
    print(dataset[cols3].describe())
    # class distribution
    print(dataset.groupby('class').size())

## Data Visualisation to understand Data
def visualiseDataset(dataset):
    cols2 = ['HSCODE','is_coc','paid_amount']
    cols3 = ['cargo_weight','teu','paid_amount'] 
    
    # box and whisker plots
    dataset[cols2].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()
    dataset[cols3].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()
    # histograms
    dataset[cols2].hist()
    plt.show()
    dataset[cols3].hist()
    plt.show()
    # scatter plot matrix
    sns.scatterplot(dataset[cols2])
    plt.show()
    sns.scatterplot(dataset[cols3])
    plt.show()