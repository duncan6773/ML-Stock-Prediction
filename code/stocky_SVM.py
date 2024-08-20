import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
class stocks_svm:
    # A SVM Object made to work directly with our data set 
    def __init__(self):
        #Initializing objects 
        self.mySvm = svm.SVC(gamma=0.0000001,)
        self.scalor = StandardScaler()
        #The SVM Fit 
        self.myFit = []
        #Pandas DataBases
        self.myData = []
        self.myLabel = []
    
    def fit_stock_data(self,data,label):
        #Note! rows with NAN will be deleted 
        #Takes in the data and label. Converts it to a numpy array and the n standardizes the data
        #Fianlly it fits an SVM to the data
        #Inputs:
        #   Data:
        #       A N*D Pandas base
        #   Label:
        #       A N*1 Pandas database
        #removing NaN rows 
        myMask = data.isnull().sum(axis = 1)
        myMask = myMask == 0
        data = data[myMask]
        label = label[myMask]
        #Adding the data to our object
        self.myData = data
        if "Ticker" in data.columns:
            data = data.drop(['Ticker'], axis =1)
        self.myLabel = label
        #Converting to Numpy
        data = data.to_numpy()
        label = label.to_numpy()
        #Normalizing the data
        self.scalor.fit(data)
        myData = self.scalor.transform(data)
        #Fitting the SVM
        self.myFit = self.mySvm.fit(myData,label[:,0])

    def predict_data(self, testData):
        myMask = testData.isnull().sum(axis = 1)
        myMask = myMask == 0
        testData = testData[myMask]
        if "Ticker" in testData.columns:
            testData = testData.drop(['Ticker'], axis =1)
        testData  = testData.to_numpy()
        myTest = self.mySvm.predict(testData)
        return(myTest)

    def findKendallCoeff(self,data,labels):
        #Inputs:
        #   Data:
        #       A N * DPandas Database That can NaN values 
        #Returns:
        #       A [D,~] numpy array of kendall coefficents 
        #Finds the Kendall coeff of the columns. This is considered to be how each 
        # feature relates to a relation to the label. If it is one, that is considered a strong 
        #positive coorelation and -1 represents a strong negative coorelation. For data selection 
        #purposes I would recomend taking the absolute value.
        # Note this ignores the NaN values and requires binary
        #classification of the labels being either zero or one
        if "Ticker" in data.columns:
            data = data.drop(['Ticker'], axis =1)
        myMask = data.isnull()
        D = myMask.shape[1]
        """
        myMask = myMask.sum(axis = 1)
        myMask = myMask == 0
        data = data[myMask]"""
        
        #Converting our data into a numpy array
        #myData = data.to_numpy()
        myKendall = np.ones([D])
        myPVal = np.ones([D])
        for i in range(D):
            #For Each Feature 
            myFeat = data.iloc[:,i]
            #Remove the null values
            myMask =  myFeat.isnull() == 0
            myFeat = myFeat[myMask]
            myLabel = labels[myMask]
            myFeat = myFeat.to_numpy()
            myLabel = myLabel.to_numpy()
            myKendall[i],myPVal[i] = kendalltau(myFeat,myLabel)
        return myKendall
    
    def findPMV(self, data):
        #Inputs:
        #   data - a N*D pandas dataframe with nan values
        #Output:
        #   PMV - a Pandas Dataframe with each value represents the PMV of the cooresponding 
        #           feature
        #finds the percent missing value or the number NaN in the features of the data

        if "Ticker" in data.columns:
            data = data.drop(['Ticker'], axis =1)
        N = data.shape[0]
        nullData = data.isnull().sum(axis=0)
        PMV = nullData / N
        return(PMV)

    def calcPercision(self,yActual, yPredict ):
        #Calculates the percision of the model
        return (precision_score(yActual, yPredict, average= 'binary', zero_division='warn'))
    def calcAccuracy(self,yActual, yPredict ):
        return (accuracy_score(yActual, yPredict))



