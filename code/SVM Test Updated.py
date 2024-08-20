import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from stocky_SVM import stocks_svm
from sklearn.model_selection import KFold

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


all_common_companies = list(set.intersection(ticker_set_2014, ticker_set_2015, ticker_set_2016, ticker_set_2017,
                                        ticker_set_2018))
#filter dataframes using the list of common companies we created above
data_2014 = data_2014[data_2014["Ticker"].isin(all_common_companies)]
data_2015 = data_2015[data_2015["Ticker"].isin(all_common_companies)]
data_2016 = data_2016[data_2016["Ticker"].isin(all_common_companies)]
data_2017 = data_2017[data_2017["Ticker"].isin(all_common_companies)]
data_2018 = data_2018[data_2018["Ticker"].isin(all_common_companies)]
new_2014 = data_2014[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio','Graham Net-Net']]
new_2015 = data_2015[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio','Graham Net-Net']]
new_2016 = data_2016[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio','Graham Net-Net']]
new_2017 = data_2017[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio','Graham Net-Net']]
new_2018 = data_2018[["Ticker",'EPS','EBIT Margin', 'Net Profit Margin','Net Cash/Marketcap','priceBookValueRatio','priceEarningsRatio', 'priceCashFlowRatio',
 'returnOnAssets','inventoryTurnover','cashFlowToDebtRatio','totalDebtToCapitalization','Earnings Yield','Debt to Assets', 'Current ratio','Graham Net-Net']]

#extracting some labels
labels_2014 = data_2014[['Class']]
labels_2015 = data_2015[['Class']]
labels_2016 = data_2016[['Class']]
labels_2017 = data_2017[['Class']]
labels_2018 = data_2018[['Class']]



totalData = new_2014.append(new_2015,sort=False)
"""
totalData = totalData.append(new_2016,sort=False)
totalData = totalData.append(new_2017,sort=False)
totalData = totalData.append(new_2018,sort=False)"""
totalLabels = labels_2014.append(labels_2015,sort=False)
"""
totalLabels = totalLabels.append(labels_2016,sort=False)
totalLabels = totalLabels.append(labels_2017,sort=False)
totalLabels = totalLabels.append(labels_2018,sort=False)"""


mySVM = stocks_svm()
"""kenCo = mySVM.findKendallCoeff(new_2014,myLabels)
kenCo = np.sqrt(kenCo * kenCo)
pmv = mySVM.findPMV(new_2014)
myDataSelect = kenCo / pmv"""
mySplitter = KFold(n_splits=5)
length = new_2014.shape[1]
testErrorSVM = np.zeros([length-1]) 
trainErrorSVM = np.zeros([length-1]) 
#Sorting based on which is the lowest
nullItter = mySVM.findPMV(new_2014).sort_values(axis = 0)
#If we want highest first ascending = false

#Now need to get the column label
colLabel = nullItter.index
len = colLabel.size

totErrorSVM = np.zeros([len-1]) 
totErrorTrain = np.zeros([len-1]) 
myPercision = np.zeros([len-1])
myAccuracy = np.zeros([len-1])
myData = totalData
myLabels = totalLabels

for i in range(len-1):
    print(i)
    #Getting out how many labels we want 
    myFeat = colLabel[1:i+2]
    myDataSet = myData[myFeat]
    #Now need to remove the rows with NaN 
    myMask = myDataSet.isnull().sum(axis = 1)
    myMask = myMask == 0
    myDataSet = myDataSet[myMask]
    myData = myData[myMask]
    myLabels = myLabels[myMask]
    foldErrorSVM = 0
    foldErrorTrain = 0
    #Gonna split the data into K folds and train an svm model 
    for trainIdx,testIdx in mySplitter.split(myDataSet):
        #Note convert your pandas data frame into a numpy array 
        #
        trainData = myDataSet.iloc[trainIdx]
        testData = myDataSet.iloc[testIdx]
        trainLabel = myLabels.iloc[trainIdx]
        testLabel = myLabels.iloc[testIdx]
        #Need to standardize data set and fit the SVM
        mySVM.fit_stock_data(trainData,trainLabel)
        #Predicting the Test Data 
        myTestSVM = mySVM.predict_data(testData)
        myTrainSVM =  mySVM.predict_data(trainData)
        #Using Euclidean distance to get the fold error
        myErrorSVM = myTestSVM != testLabel.to_numpy()
        myErrorSVM = np.sum(myErrorSVM)
        

        myErrorTrain = myTrainSVM != trainLabel.to_numpy()
        myErrorTrain = np.sum(myErrorSVM)



        #note we need to divide by the number of data points each time because each loop 
        #the number of data points is getting smaller 
        foldErrorSVM = foldErrorSVM + (myErrorSVM/ testData.shape[0])
        foldErrorTrain = foldErrorTrain + (myErrorTrain/ trainData.shape[0])
    myAccuracy[i] = mySVM.calcAccuracy(testLabel.to_numpy(),myTestSVM)  
    myPercision[i] = mySVM.calcPercision(testLabel.to_numpy(),myTestSVM)
    totErrorSVM[i]= foldErrorSVM
    totErrorTrain[i] = foldErrorTrain
    nullItter = mySVM.findPMV(myData).sort_values(axis = 0)
    #nullItter = nullItter / mySVM.findKendallCoeff()
    colLabel = nullItter.index

#Lets plot how the error changes
plt.plot(np.arange(len-1) + 1,totErrorSVM,'b')
plt.plot(np.arange(len-1) + 1,totErrorTrain,'r')
plt.xlabel('Number of Features')
plt.ylabel('Error ')
plt.legend(['SVM Test Error', 'SVM Train Error'])
plt.title("Error of SVM with Five Fold Cross Validation")
plt.show()
plt.figure()
plt.plot(np.arange(len-1) + 1,myPercision)
plt.xlabel('Number of Features')
plt.ylabel('Percision ')
plt.title("Percision of SVM with Five Fold Cross Validation")
plt.show()
plt.figure()
plt.plot(np.arange(len-1) + 1,myAccuracy)
plt.xlabel('Number of Features')
plt.ylabel('Accuracy ')
plt.title("Accuracy of SVM with Five Fold Cross Validation")
plt.show()