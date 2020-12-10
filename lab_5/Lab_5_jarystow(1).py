#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
from sklearn.preprocessing import KBinsDiscretizer as KBD
from sklearn.model_selection import train_test_split as TTS
from sklearn.base import BaseEstimator as BE, ClassifierMixin as CM
from collections import defaultdict, Counter
from sklearn.naive_bayes import GaussianNB as GNB
import pandas as pd


# In[2]:


#1.
#repo UVI | dataset: wine

wineRaw = np.genfromtxt("wine.data",delimiter=',')

X = wineRaw[:,1:] #args 178x13
y = wineRaw[:,0] #class 178X1


# In[3]:


#2.
NBINS =3
est = KBD(n_bins=NBINS , encode='ordinal', strategy='kmeans')
Xt =est.fit_transform(X)
print(X)
print(Xt)


# In[4]:


#3.
X_train, X_test, y_train, y_test = TTS( Xt, y, test_size=0.3, random_state=42)


# In[5]:


#4.
class NBC_discrete(BE, CM):
    def __init__(self,laPlace_switch:bool =False, nBuckets:int =3):
        self.laPlace = laPlace_switch
        self.nBuckets = nBuckets
        
    def __stackSeparators(self):
        for key in self.separator.keys():
            self.separator[key]=np.vstack( self.separator[key] ) 
    def __separateByClass(self): 
        self.separator=defaultdict(list)
        for v, k in zip(self.X,self.y):
            self.separator[int(k)].append(np.array(v,dtype=int))
        self.__stackSeparators()

    def __aPrioriClass(self): # rozkład a priori klas P(Y =y)
        self.aPriori ={}
        for key in self.separator.keys():
            self.aPriori[key] = len(self.separator[key])/self.size[0]
            
    def __sizeOfClasses(self):
        self.sizeOfSeparator ={}
        for key in self.separator.keys():
            self.sizeOfSeparator[key] = len(self.separator[key])
      
    def __fillValues(self, counter):
        for attribute in range(0, self.nBuckets):
            if attribute not in counter.keys():
                counter[attribute] = 0.0
        
    def __conditionalDistribution(self): 
        self.__numOfAtributes = int(self.size[1])
        minY = int(min(self.y))
        maxY = int(max(self.y)+1)
        self.conditionalDistDict = [list(range(0,self.__numOfAtributes))for x in range(minY,maxY) ]
        for key in self.separator.keys():
            dictKey = key - 1
            for attribute in range(0, self.__numOfAtributes):
                c = Counter(self.separator[key][:,attribute])
                self.__fillValues(c)
                self.conditionalDistDict[dictKey][attribute] ={}
                for value in c.keys():
                    if not self.laPlace: # zastosowanie przełącznika poprawki LaPlace'a
                        self.conditionalDistDict[dictKey][attribute][value] = c[value]/self.sizeOfSeparator[key]
                    else:
                        self.conditionalDistDict[dictKey][attribute][value] =  (c[value]+1)/(self.sizeOfSeparator[key]+ len(c.keys()))
    
    def __setProbabilities(self):
        self.__separateByClass()
        self.__aPrioriClass()
        self.__sizeOfClasses()
        self.__conditionalDistribution()
        
    def __calculateKeyLikelihood(self, X_row):
        yLikelihood = np.zeros(len(self.separator.keys()))
        for key in self.separator.keys():
            keyProbability =1
            dictKey = key -1
            for attribute,value in enumerate(X_row):
                tempProb = self.conditionalDistDict[dictKey][attribute][value] 
                keyProbability *=tempProb
            yLikelihood[dictKey]=keyProbability*self.aPriori[key]
        return yLikelihood

    def fit(self,X,y):
        self.size = np.shape(X)
        self.X =X
        self.y = y
        self.__setProbabilities()
        
    def predict(self,X):
        yPredicted = []
        for X_row in X:
            yLikelihood = self.__calculateKeyLikelihood( X_row)
            yPredicted.append(np.argmax(yLikelihood, axis=0) +1) # +1 bo zakres etykiet 1-3 
        return np.transpose(yPredicted)
    
    def predict_proba(self,X):
        yPredicted = []
        for X_row in X:
            yLikelihood = self.__calculateKeyLikelihood(X_row)
            arg = np.argmax(yLikelihood, axis=0)
            yProbability = yLikelihood[arg]/ np.sum(yLikelihood)
            yPredicted.append(yProbability)
        return np.transpose(yPredicted)


# In[6]:


def accuracy_score(y, yPredicted):
    return (np.sum(y ==yPredicted)/np.shape(y)[0])*100


# In[7]:


def printResults(yPredicted, yPredictedProbability, y, stringSet:str= None, LaPlaceEnabled:bool = False):
    stringLaPlace = "Enabled" if LaPlaceEnabled else "Disabled"
    #print(yPredicted)
    #print(y)
    print("--------------------------------------------------------------------------------------")
    print("{} set ".format(stringSet)) #Train | Test
    print("LaPlace's correction {}".format(stringLaPlace))
    print("Pobabilities of predicted y:")
    print(yPredictedProbability)
    print("Accuracy of {} set: {}%\n\n".format(stringSet, str(accuracy_score(y, yPredicted))) )


# In[8]:


# eksperyment bez poprawki LaPlace'a
LaPlaceEnabled = False

discNBC = NBC_discrete(LaPlaceEnabled,NBINS)
discNBC.fit(X_train, y_train)


#-------------------------------------------TEST SET-------------------------------------------
yPredicted = discNBC.predict(X_test)
yPredictedProbability = discNBC.predict_proba(X_test)
printResults(yPredicted, yPredictedProbability, y_test, stringSet= "Test", LaPlaceEnabled= LaPlaceEnabled)
#----------------------------------------TRAIN SET----------------------------------------------
yPredicted = discNBC.predict(X_train)
yPredictedProbability = discNBC.predict_proba(X_train)
printResults(yPredicted, yPredictedProbability, y_train, stringSet= "Train", LaPlaceEnabled= LaPlaceEnabled)


# In[9]:


# eksperyment z poprawką LaPlace'a

LaPlaceEnabled = True
discNBC = NBC_discrete(LaPlaceEnabled,NBINS)
discNBC.fit(X_train, y_train)


#-------------------------------------------TEST SET-------------------------------------------
yPredicted = discNBC.predict(X_test)
yPredictedProbability = discNBC.predict_proba(X_test)
printResults(yPredicted, yPredictedProbability, y_test, stringSet= "Test", LaPlaceEnabled= LaPlaceEnabled)
#----------------------------------------TRAIN SET----------------------------------------------
yPredicted = discNBC.predict(X_train)
yPredictedProbability = discNBC.predict_proba(X_train)
printResults(yPredicted, yPredictedProbability, y_train, stringSet= "Train", LaPlaceEnabled= LaPlaceEnabled)


# In[44]:



#############################################################################################################
#-------------------------------------------continuous NBC---------------------------------------------------
#wersja bezpieczna numerycznie 
#4.
class NBC_continuous(BE, CM):


        
    def __stackSeparators(self):
        for key in self.separator.keys():
            self.separator[key]=np.vstack( self.separator[key] ) 
    def __separateByClass(self): 
        self.separator=defaultdict(list)
        for v, k in zip(self.X,self.y):
            self.separator[int(k)].append(np.array(v))
        self.__stackSeparators()
        
    def __aPrioriClass(self): # rozkład a priori klas P(Y =y)
        self.aPriori ={}
        for key in self.separator.keys():
            self.aPriori[key] = len(self.separator[key])/self.size[0]
            
    def __calculateAvg(self):
        self.__numOfAtributes = int(self.size[1])
        self.__minY = int(min(self.y))
        self.__maxY = int(max(self.y)+1)
        self.avgDict = [list(range(0,self.__numOfAtributes))for x in range(self.__minY,self.__maxY) ]
        for key in self.separator.keys():
            dictKey = key - 1
            for attribute in range(0, self.__numOfAtributes):
                #3print(self.separator[key][:,attribute])
                self.avgDict[dictKey][attribute] = np.mean(self.separator[key][:,attribute],axis =0)
                #print(self.avgDict[dictKey][attribute])
            
    def __calculateStd(self):

        self.stdDict = [list(range(0,self.__numOfAtributes))for x in range(self.__minY,self.__maxY) ]
        for key in self.separator.keys():
            dictKey = key - 1
            for attribute in range(0, self.__numOfAtributes):
                self.stdDict[dictKey][attribute] = np.std(self.separator[key][:,attribute],axis =0, ddof = 1)
      #      print(self.stdDict[dictKey])
        

    def __setAvgStd(self):
        self.__separateByClass()
        self.__aPrioriClass()
        self.__calculateAvg()
        self.__calculateStd()
        
    def __calculateKeyLikelihood(self, X_row):
        yLikelihood = np.zeros(len(self.separator.keys()))
        for key in self.separator.keys():
            keyProbability =0
            dictKey = key -1
            for attribute,value in enumerate(X_row):
                tempDiv =  (value - self.avgDict[dictKey][attribute])**2/(2* self.stdDict[dictKey][attribute]**2)
                tempLog = -np.log(self.stdDict[dictKey][attribute])
                tempProb = tempLog - tempDiv
                keyProbability +=tempProb
            yLikelihood[dictKey]=keyProbability+np.log(self.aPriori[key])
        return yLikelihood

    def fit(self,X,y):
        self.size = np.shape(X)
        self.X =X
        self.y = y
        self.__setAvgStd()
        
    def predict(self,X):
        yPredicted = []
        for X_row in X:
            yLikelihood = self.__calculateKeyLikelihood( X_row)
            yPredicted.append(np.argmax(yLikelihood, axis=0) +1) # +1 bo zakres etykiet 1-3 
        return np.transpose(yPredicted)
    
    def predict_proba(self,X):
        yPredicted = []
        for X_row in X:
            yLikelihood = self.__calculateKeyLikelihood(X_row)
            arg = np.argmax(yLikelihood, axis=0)
            yProbability = yLikelihood[arg]/ np.sum(yLikelihood)
            yPredicted.append(yProbability)
        return np.transpose(yPredicted)


# In[37]:


X_train, X_test, y_train, y_test = TTS( X, y, test_size=0.3, random_state=42)


# In[39]:


for X_row in X_train:
    print(X_row)


# In[49]:


def printResults2(yPredicted, yPredictedProbability, y, stringSet:str= None):

    print("--------------------------------------------------------------------------------------")
    print("{} set ".format(stringSet)) #Train | Test
    print("Pobabilities of predicted y:")
    print(yPredictedProbability)
    print(y)
    print(yPredicted)
    print("Accuracy of {} set: {}%\n\n".format(stringSet, str(accuracy_score(y, yPredicted))) )


# In[51]:




countNBC = NBC_continuous()
countNBC.fit(X_train, y_train)


#-------------------------------------------TEST SET-------------------------------------------
yPredicted = countNBC.predict(X_test)
yPredictedProbability = countNBC.predict_proba(X_test)
printResults2(yPredicted, yPredictedProbability, y_test, stringSet= "Test")
#----------------------------------------TRAIN SET----------------------------------------------
yPredicted = countNBC.predict(X_train)
yPredictedProbability = countNBC.predict_proba(X_train)
printResults2(yPredicted, yPredictedProbability, y_train, stringSet= "Train")


# In[57]:


#Gaussian NBC
countNBC = GNB()
countNBC.fit(X_train, y_train)


#-------------------------------------------TEST SET-------------------------------------------
yPredicted = countNBC.predict(X_test)
yPredictedProbability = countNBC.predict_proba(X_test)
printResults2(yPredicted, yPredictedProbability, y_test, stringSet= "Test")
#----------------------------------------TRAIN SET----------------------------------------------
yPredicted = countNBC.predict(X_train)
yPredictedProbability = countNBC.predict_proba(X_train)
printResults2(yPredicted, yPredictedProbability, y_train, stringSet= "Train")


# In[ ]:




