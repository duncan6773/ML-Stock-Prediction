import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stocky_SVM import stocks_svm
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from copy import deepcopy

#convert all 
data_2014 = pd.read_csv("/Users/dunca/Documents/ML/ProjectData/2014_Financial_Data.csv")
data_2015 = pd.read_csv("/Users/dunca/Documents/ML/ProjectData/2015_Financial_Data.csv")
data_2016 = pd.read_csv("/Users/dunca/Documents/ML/ProjectData/2016_Financial_Data.csv")
data_2017 = pd.read_csv("/Users/dunca/Documents/ML/ProjectData/2017_Financial_Data.csv")
data_2018 = pd.read_csv("/Users/dunca/Documents/ML/ProjectData/2018_Financial_Data.csv")

#the companies are denoted by their stock ticker but the ticker column is unnamed
#this cell renames that to "Ticker" for all dataframes
data_2014.rename( columns={'Unnamed: 0':'Ticker'}, inplace=True )
data_2015.rename( columns={'Unnamed: 0':'Ticker'}, inplace=True )
data_2016.rename( columns={'Unnamed: 0':'Ticker'}, inplace=True )
data_2017.rename( columns={'Unnamed: 0':'Ticker'}, inplace=True )
data_2018.rename( columns={'Unnamed: 0':'Ticker'}, inplace=True )


#construct 5 different sets of ticker labels, 1 for eachh data frame
ticker_set_2014 = set(data_2014["Ticker"])
ticker_set_2015 = set(data_2015["Ticker"])
ticker_set_2016 = set(data_2016["Ticker"])
ticker_set_2017 = set(data_2017["Ticker"])
ticker_set_2018 = set(data_2018["Ticker"])


#Defining a function that will remove all the rows with an nan value
def removeNaN(data,labels,myMask = np.zeros(0)):
    #Inputs: 
    #       data:
    #           A Pandas data frame with our features and samples
    #       labels:
    #           A Pandas data frame that is our labels
    N = data.shape[0]
    #Getting the percentage of null values for the features
    nullData = data.isnull().sum(axis=0)
    PMV = nullData / N
    #If the Percent mean value is over 0.3, remove those features 
    if myMask.size == 0:
        myMask = PMV < 0.3
    data = data[myMask.index[myMask]]
    #If the row has a null value
    nullData = data.isnull().sum(axis=1)
    myMasky = nullData == 0
    data = data[myMasky]
    labels = labels[myMasky]
    return data,labels,myMask


all_common_companies = list(set.intersection(ticker_set_2014, ticker_set_2015, ticker_set_2016, ticker_set_2017,
                                        ticker_set_2018))
#filter dataframes using the list of common companies we created above
data_2014 = data_2014[data_2014["Ticker"].isin(all_common_companies)]
data_2015 = data_2015[data_2015["Ticker"].isin(all_common_companies)]
data_2016 = data_2016[data_2016["Ticker"].isin(all_common_companies)]
data_2017 = data_2017[data_2017["Ticker"].isin(all_common_companies)]
data_2018 = data_2018[data_2018["Ticker"].isin(all_common_companies)]

#Filtering out features that are not dependant on currency 
new_2014 = data_2014[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio',
'Free Cash Flow margin']]
new_2015 = data_2015[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio',
'Free Cash Flow margin']]
new_2016 = data_2016[["Ticker",'EPS','EBIT Margin','Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio',
'Free Cash Flow margin']]
new_2017 = data_2017[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio',
'Free Cash Flow margin']]
new_2018 = data_2018[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio',
'Free Cash Flow margin']]


#extracting some labels
labels_2014 = data_2014[['Class']]
labels_2015 = data_2015[['Class']]
labels_2016 = data_2016[['Class']]
labels_2017 = data_2017[['Class']]
labels_2018 = data_2018[['Class']]


#construct 5 different sets of ticker labels, 1 for eachh data frame
ticker_set_2014 = set(data_2014["Ticker"])
ticker_set_2015 = set(data_2015["Ticker"])
ticker_set_2016 = set(data_2016["Ticker"])
ticker_set_2017 = set(data_2017["Ticker"])
ticker_set_2018 = set(data_2018["Ticker"])


#Defining a function that will remove all the rows with an nan value
def removeNaN(data,labels,myMask = np.zeros(0)):

    #Inputs: 
    #       data:
    #           A Pandas data frame with our features and samples
    #       labels:
    #           A Pandas data frame that is our labels
    N = data.shape[0]
    #Getting the percentage of null values for the features
    nullData = data.isnull().sum(axis=0)
    PMV = nullData / N
    #If the Percent mean value is over 0.3, remove those features 
    if myMask.size == 0:
        myMask = PMV < 0.3
    data = data[myMask.index[myMask]]
    #If the row has a null value
    nullData = data.isnull().sum(axis=1)
    myMasky = nullData == 0
    data = data[myMasky]
    labels = labels[myMasky]
    return data,labels,myMask


myData2014,myLabels2014,_ = removeNaN(data_2014,labels_2014)
myData2015,myLabels2015,_ = removeNaN(data_2015,labels_2015)
myData2016,myLabels2016,_ = removeNaN(data_2016,labels_2016)
myData2018,myLabels2017,_ = removeNaN(data_2018,labels_2018)


all_common_companies = list(set.intersection(ticker_set_2014, ticker_set_2015, ticker_set_2016, ticker_set_2017,
                                        ticker_set_2018))
#filter dataframes using the list of common companies we created above
data_2014 = data_2014[data_2014["Ticker"].isin(all_common_companies)]
data_2015 = data_2015[data_2015["Ticker"].isin(all_common_companies)]
data_2016 = data_2016[data_2016["Ticker"].isin(all_common_companies)]
data_2017 = data_2017[data_2017["Ticker"].isin(all_common_companies)]
data_2018 = data_2018[data_2018["Ticker"].isin(all_common_companies)]

#Filtering out features that are not dependant on currency 
new_2014 = data_2014[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio',
'Free Cash Flow margin']]
new_2015 = data_2015[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio',
'Free Cash Flow margin']]
new_2016 = data_2016[["Ticker",'EPS','EBIT Margin','Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio',
'Free Cash Flow margin']]
new_2017 = data_2017[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio',
'Free Cash Flow margin']]
new_2018 = data_2018[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio',
'Free Cash Flow margin']]


#extracting some labels
labels_2014 = data_2014[['Class']]
labels_2015 = data_2015[['Class']]
labels_2016 = data_2016[['Class']]
labels_2017 = data_2017[['Class']]
labels_2018 = data_2018[['Class']]

#Getting out the data and labels we are going to train our model on 

totalData = new_2015
#totalData = new_2014.append(new_2015,sort=False)
#totalData = totalData.append(new_2016,sort=False)
#totalData = totalData.append(new_2017,sort=False)
totalData = totalData.append(new_2018,sort=False)

totalLabels = labels_2015
#totalLabels = labels_2014.append(labels_2015,sort=False)
#totalLabels = totalLabels.append(labels_2016,sort=False)
#totalLabels = totalLabels.append(labels_2017,sort=False)
totalLabels = totalLabels.append(labels_2018,sort=False)

#Defining some our variable 
scalor = StandardScaler()
mySplitter = KFold(n_splits=5)
length = totalData.shape[1]
myDataSet = totalData
## 
#max depth is a hyper paramater 
myForest = RandomForestClassifier(max_depth = 6)
mySvm = svm.SVC(gamma=0.00001,)
myAcc = np.zeros([2,5])
i = 0

#Removing the companies names from the data 
myDataSet,totalLabels,_ = removeNaN(myDataSet,totalLabels)
if "Ticker" in myDataSet.columns:
            myDataSet = myDataSet.drop(['Ticker'], axis =1)

#Now lets train a decision tree to find out the importances of our features 
featData = myDataSet.to_numpy()
featLabel = totalLabels.to_numpy()
myTree = DecisionTreeClassifier(max_depth=10)
myTree.fit(featData,featLabel)

#Getting out how much each feature coorelates to knowing label
myImport = myTree.feature_importances_
D = myImport.shape[0]

#Masking out the most important features 
myMask = myImport > 0.065
myCols = myDataSet.columns
myColsMasked = myCols[myMask]
featData = featData[:,myMask]
plt.bar(range(D),myImport)
plt.xlabel("Feature")
plt.ylabel("Gini Importance")
plt.title("Feature Gini Importances")
plt.xticks(ticks = range(D), labels=myCols,rotation=70)
plt.show()
plt.figure()
keptFeat = featData.shape[1]
myTree.fit(featData,featLabel)
myImport = myTree.feature_importances_
plt.bar(range(keptFeat),myImport)
plt.xlabel("Feature")
plt.ylabel("Gini Importance")
plt.title("Feature Gini Importances")
plt.xticks(ticks = range(keptFeat), labels=myColsMasked,rotation=60)
plt.show()