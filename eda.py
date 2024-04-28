# load necessary libraries
import time
import pandas as pd
import pickle as pk
import numpy as np
import os
from datetime import datetime

# clustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# predict time series
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

start_time = time.time()

# declare contants
base_dir = '~\\project\\ExploratoryDataAnalysis'
excel_file = 'aiml_test_data.xlsx'
filename = os.path.join(base_dir, excel_file)

# -------------------------------------------------------------------------
# Helper modules for Descriptive Statistics
# -------------------------------------------------------------------------
def get_redundant_pairs(df):
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

def get_top_abs_correlations(df, n=5):
        au_corr = df.corr().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

def corrank(X):
        import itertools
        df = pd.DataFrame([[(i,j),
                   X.corr().loc[i,j]] for i,j in list(itertools.combinations(X.corr(), 2))],
                   columns=['pairs','corr'])
        print(df.sort_values(by='corr',ascending=False))
        print()

# -------------------------------------------------------------------------
# load dataset
# -------------------------------------------------------------------------
def load_dataset(filename):
    dataset = pd.read_excel(filename, sheet_name='Sheet1', header=0, na_values='NaN')

    print(dataset.shape);    print(dataset.head(5));    print(dataset.columns)

    feature_names = ['port_of_loading', 'port_of_discharge', 'HSCODE', 'is_coc',
       'cargo_weight', 'expected_time_of_departure', 'teu']
    target = 'HSCODE2'

    return feature_names, target, dataset

# -------------------------------------------------------------------------
# find missing values in dataset if exists
# -------------------------------------------------------------------------
def find_missing_value(feature_names, target, dataset):
        # Count Number of Missing Value on Each Column
        print('\nCount Number of Missing Value on Each Column: ')
        print(dataset.isnull().sum(axis=0))

# -------------------------------------------------------------------------
# factorize text values & Sort by
# -------------------------------------------------------------------------
def factorzie_text_values(dataset):
    ports_of_loading, pol = pd.factorize(dataset['port_of_loading'])
    dataset['pol'] = pd.DataFrame(ports_of_loading)

    ports_of_discharge, pod = pd.factorize(dataset['port_of_discharge'])
    dataset['pod'] = pd.DataFrame(ports_of_discharge)

    dataset['is_coc'] = dataset['is_coc'].astype(int)

    date_string = dataset['expected_time_of_departure'].dt.strftime('%Y%m%d')
    dataset['date'] = date_string.astype(int)

    dataset['HSCODE2'] = (dataset['HSCODE']/10000).astype(int)

    dataset.sort_values(by=['expected_time_of_departure'], axis=0, ascending=True, inplace=True)

    print(dataset.head(5))

    return pol, pod, dataset


# -------------------------------------------------------------------------
# 1. 각 컬럼의 의미
# -------------------------------------------------------------------------

# pol 선착항 ['KRMAS: 한국 마산항', 'KRPUN: 한국 부산신항으로 추정', 'KRPUS: 한국 부산항', 'KRPTK: 한국 평택항']
# pod 도착항 ['JPNGO: 일본 나고야항', 'JPTYO: 일본 도쿄항', 'JPKNZ: 일본 카자나와항', 'JPTRG: 일본 츠루가항', 'JPOSA: 일본 오사카항']
# HSCODE: HS CODE는 국제통일 상품분류체계에 따라 대외 무역거래 상품을 총괄적으로 분류한 품목 분류 코드
#   이 HS CODE는 총 6자리로 구성되어 있으며, 우리나라는 물건의 세부 분류를 위해 4자리를 추가해 사용하고 있음
# coc(Carrier Own Container): 선사 소유의 컨테이너로 대부분의 경우 coc로 진행됨. 선사에게 정해진 시간 및 장소에서 픽업하고 반납해야 함.
# soc(Shipper Own Container): 화주 소유의 컨테이너로 시간에 의한 픽업 및 반납에 대한 제약이 없어 시간적으로 효율적이고 선착 후 일시적 창고 역할까지 가능.
# cargo_weight: 운송 화물의 무게
# expected_time_of_departure: 도착 예정 일시
# teu(twenty-foot equivalent unit): 20 피트 길이의 컨테이너 크기를 부르는 단위로 컨테이너선이나 컨테이너 부두 등에서 주로 쓰인다.
#   20 피트 표준 컨테이너의 크기를 기준으로 만든 단위로 배나 기차, 트럭 등의 운송 수단간 용량을 비교를 쉽게하기 위해 만들어졌다.
# paid_amount: 매출

# -------------------------------------------------------------------------
# descriptive statistics and correlation matrix
# -------------------------------------------------------------------------
def data_descriptiveStats(feature_names, target, dataset):
        # Count Number of Missing Value on Each Column
        print(); print('Count Number of Missing Value on Each Column: ')
        print(); print(dataset[feature_names].isnull().sum(axis=0))
        print(); print(dataset[target].isnull().sum(axis=0))

        # Get Information on the feature variables
        print(); print('Get Information on the feature variables: ')
        print(); print(dataset[feature_names].info())
        print(); print(dataset[feature_names].describe())

        # correlation
        print(); print(dataset[feature_names].corr())

        # Ranking of Correlation Coefficients among Variable Pairs
        print(); print("Ranking of Correlation Coefficients:")
        corrank(dataset[feature_names])

        # Print Highly Correlated Variables
        print(); print("Highly correlated variables (Absolute Correlations):")
        print(); print(get_top_abs_correlations(dataset[feature_names], 8))

        # Get Information on the target
        print(); print(dataset[target].describe())
        print(); print(dataset.groupby(target).size())

# -------------------------------------------------------------------------
# data visualisation and correlation graph
# -------------------------------------------------------------------------
def data_visualization(feature_names, target, dataset):
        fig, ax = plt.subplots(1,3, figsize=(11, 5))
        sns.countplot(x='port_of_loading', data=dataset, ax=ax[0])
        sns.countplot(x='port_of_discharge', data=dataset, ax=ax[1])
        sns.countplot(x='is_coc', data=dataset, ax=ax[2])
        fig.show()

        feature_names = ['cargo_weight', 'teu', 'paid_amount']
        feature_num = len(feature_names)
        # BOX plots USING box and whisker plots
        i = 1
        print(); print('BOX plot of each Numerical features')
        plt.figure(figsize=(11, 9))
        for col in feature_names:
            plt.subplot(feature_num,2,i)
            dataset[col].plot(kind='box', subplots=True, sharex=False, sharey=False)
            i += 1
        plt.show()

        # USING histograms
        j = 1
        print(); print('Histogram of each Numerical Feature')
        plt.figure(figsize=(11, 9))
        for col in feature_names:
            plt.subplot(feature_num,2,j)
            dataset[col].hist()
            j += 1
        plt.show()

        feature_names = ['pol', 'pod', 'HSCODE', 'is_coc', 'cargo_weight', 'date', 'teu', 'paid_amount']
        feature_num = len(feature_names)
        # correlation matrix
        print(); print('Correlation Matrix of All Numerical Features')
        fig = plt.figure(figsize=(11,9))
        ax = fig.add_subplot(111)
        cax = ax.matshow(dataset[feature_names].corr(), vmin=-1, vmax=1, interpolation='none')
        fig.colorbar(cax)
        ticks = np.arange(0,feature_num,1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks, labels=feature_names)
        plt.show()

        # Correlation Plot using seaborn
        print(); print("Correlation plot of Numerical features")
        # Compute the correlation matrix
        corr = dataset[feature_names].corr()
        print(corr)
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin= -1.0, center=0, square=True,
                    linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()

        # PairPlot using seaborn
        print(); print('Scatter Matrix Plot')
        sns.pairplot(dataset, hue='HSCODE2')
        plt.show()

        # Pie chart for Categorical Variables
        print(); print('PIE Chart of for Target: ')
        plt.figure(figsize=(11,9))
        i = 1
        target = ['port_of_loading', 'port_of_discharge', 'is_coc']
        for colName in target:
            labels = []; sizes = []
            df = dataset.groupby(colName).size()
            for key in df.keys():
                labels.append(key)
                sizes.append(df[key])
            # Plot PIE Chart with %
            plt.subplot(2,2,i)
            plt.axis('on')
            plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
                            labelleft=True, labeltop=True, labelright=False, labelbottom=False)
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
            plt.axis('equal')
            i += 1
            # plt.savefig('Piefig.pdf', format='pdf')
        plt.show()

        # paid_amount in time series
        plt.figure(figsize=(11,7))
        plt.plot(dataset['expected_time_of_departure'], dataset['paid_amount'])
        plt.title("Paid Amount in time series")
        plt.xlabel("Date")
        plt.show()

# -------------------------------------------------------------------------
# 2.1 군집화
# -------------------------------------------------------------------------

# 2.1.1 군집화 방법 비교 - K-Means, Mean Shift, DBSCAN, Agglomerative Hierarchical Clustering

# set data and PCA
def pca():
    feature_names = ['is_coc', 'cargo_weight', 'teu', 'paid_amount']
    X = dataset[feature_names]
    X = StandardScaler().fit_transform(X)
    # 4개의 속성을 2차원 평면에 그리기 위해 PCA로 2개로 차원 축소
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X)

    dataset['pca_x'] = pca_transformed[:,0]
    dataset['pca_y'] = pca_transformed[:,1]

    return X

# 2.1.1.1 KMeans
def cluster_KMeans(n_clusters, X):
    km = KMeans(n_clusters=n_clusters, init='k-means++')
    km.fit_transform(X)
    dataset['kmcluster']= km.labels_
    print('K-Means')
    print(km.labels_)

    for i in range(0,n_clusters-1):
        marker_ind = dataset[dataset['kmcluster']==i].index

        plt.scatter(x=dataset.loc[marker_ind,'pca_x'], y=dataset.loc[marker_ind,'pca_y'])

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('10 Clusters Visualization by K-Means')
    plt.show()

# 2.1.1.2 DBSCAN
def cluster_DBSCAN(n_clusters, X):
    dbscan = DBSCAN(eps=0.5, min_samples=n_clusters)
    dbscan.fit(X)

    dataset['dbscancluster']= dbscan.labels_
    print('DBSCAN')
    print(dbscan.labels_)

    for i in range(0,n_clusters-1):
        marker_ind = dataset[dataset['dbscancluster']==i].index

        plt.scatter(x=dataset.loc[marker_ind,'pca_x'], y=dataset.loc[marker_ind,'pca_y'])

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('10 Clusters Visualization by DBSCAN')
    plt.show()

# 2.1.1.3 Agglogmerative Clustering : 가장 가까운 2개부터 묶어보면서 거리를 늘려가는 방식
def cluster_Agglogmerative(n_clusters, X):
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg.fit(X)

    dataset['aggcluster']= agg.labels_
    print('Agglogmerative')
    print(agg.labels_)

    for i in range(0,n_clusters-1):
        marker_ind = dataset[dataset['aggcluster']==i].index

        plt.scatter(x=dataset.loc[marker_ind,'pca_x'], y=dataset.loc[marker_ind,'pca_y'])

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('10 Clusters Visualization by Agglogmerative Hierarchical Clustering')
    plt.show()


# 2.1.3 군집화 결과에 대한 해석
# 크게 2개의 cluster로 나뉘는 모습은 Container 유무(is_coc)에 따른 paid_amount의 차이로 보이며
# 각 cluster가 선형을 보이는 모습은 teu에 따른 paid_amount의 관계로 보임

# -------------------------------------------------------------------------
# 2.2 시계열 예측
# -------------------------------------------------------------------------

# ACF and PACF
def acf_pacf():
    f = plt.figure(figsize=(11,9))
    ax1 = f.add_subplot(211)
    ax1.set_title('time series of paid amount')
    ax1.plot(ts)

    ax2 = f.add_subplot(223)
    plot_acf(ts, ax=ax2)
    ax3 = f.add_subplot(224)
    plot_pacf(ts, ax=ax3)
    plt.show()

    # find diff - 1st order differencing
    f = plt.figure(figsize=(11,9))
    ax11 = f.add_subplot(211)
    ax11.set_title('1nd Order Differencing')
    ax11.plot(ts.diff())

    ax12 = f.add_subplot(223)
    plot_acf(ts.diff().dropna(), ax=ax12)
    ax13 = f.add_subplot(224)
    plot_pacf(ts.diff().dropna(), ax=ax13)
    plt.show()

    # 2nd order differencing
    f = plt.figure(figsize=(11,9))
    ax21 = f.add_subplot(211)
    ax21.set_title('2nd Order Differencing')
    ax21.plot(ts.diff().diff().dropna())

    ax22 = f.add_subplot(223)
    plot_acf(ts.diff().diff().dropna(), ax=ax22)
    ax23 = f.add_subplot(224)
    plot_pacf(ts.diff().diff().dropna(), ax=ax23)
    plt.show()


def adf():
    result = adfuller(ts)
    print('p-value: ', result[1])

    result = adfuller(ts.diff().dropna())
    print('p-value: ', result[1])

    result = adfuller(ts.diff().diff().dropna())
    print('p-value: ', result[1])

# ARIMA LIBRARY

def build_model(ts):
    # fit model
    model = ARIMA(ts[ts.index < datetime(2024,4,1)], order=(1,1,2))
    model_fit = model.fit()

    # summary of fit model
    print(model_fit.summary())

    # predict
    forecast = model_fit.predict(start="2024-04-01", end="2024-04-30")

    # visualization
    plt.figure(figsize=(22,10))
    plt.plot(ts, label = "original")
    plt.plot(forecast, label = "predicted")
    plt.title("expected_time_of_departure")
    plt.xlabel("Date")
    plt.legend()
    plt.show()

# predict all path
def predict_model(ts):
    # fit model
    model2 = ARIMA(ts, order=(1,1,2)) # (ARMA) = (1,0,1)
    model_fit2 = model2.fit()
    forecast2 = model_fit2.predict()
    error = mean_squared_error(ts, forecast2)
    print("error: " ,error)
    # visualization
    plt.figure(figsize=(22,10))
    plt.plot(ts, label = "original")
    plt.plot(forecast2,label = "predicted")
    plt.title("Time Series Forecast")
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.legend()
    plt.savefig('graph.png')

    plt.show()
    return model2

# ------------------
# save the model
# ------------------
def save_model(model):
        with open('paid_amount_model.pickle', 'wb') as f:
            pk.dump(model, f)


# ------------------------------------------------
# Load the model from disk and make predictions
# ------------------------------------------------
def final_prediction(feature_names, filename):
        # load model
        f = open('paid_amount_model.pickle', 'rb')
        model = pk.load(f); f.close()

        # load dataset
        dataset = pd.read_excel(filename, sheet_name='Sheet1', header=0, na_values='NaN')



if __name__ == '__main__':  
    # execute the function
    feature_names, target, dataset = load_dataset(filename)
    find_missing_value(feature_names, target, dataset)
    pol, pod, dataset = factorzie_text_values(dataset)
    
    feature_names = ['pol', 'pod', 'HSCODE', 'is_coc', 'cargo_weight', 'date', 'teu', 'paid_amount']
    data_descriptiveStats(feature_names, target, dataset)

    data_visualization(feature_names, target, dataset)
    
    # 2.1 군집화
    n_clusters = 6
    X = pca()
    cluster_KMeans(n_clusters, X)
    cluster_DBSCAN(n_clusters, X)
    cluster_Agglogmerative(n_clusters, X)

    # 2.1.3 군집화 결과에 대한 해석
    # 크게 2개의 cluster로 나뉘는 모습은 Container 유무(is_coc)에 따른 paid_amount의 차이로 보이며 
    # 각 cluster가 선형을 보이는 모습은 teu에 따른 paid_amount의 관계로 보임

    # 2.2 시계열 예측
    ts = dataset.groupby('expected_time_of_departure')['paid_amount'].sum()
    acf_pacf()
    adf()   
    build_model(ts)
    model = predict_model(ts)
    save_model(model)
    final_prediction(feature_names, filename)    

    print()
    print("Required Time %s seconds: " % (time.time() - start_time))