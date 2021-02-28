#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MaxNLocator
get_ipython().magic('matplotlib inline')


# In[2]:


class Chessboard:
    def __init__(self,n=4,board_state=None): # inicjalizacja
        self.size = n # do w
        self.board = np.zeros((self.size),dtype=np.int) if board_state is None else board_state  
            
    def getPositions(self):
        return  self.board[:]      
        
    def getNumberOfAttacks(self):
        numberOfAttacks =0
        numberOfAttacks += self.__diagonalAttacks()
        return numberOfAttacks
    
    def CHECK(self): # test poprawności wektora
        counter       = Counter(self.board)
        for value in counter.values():
            if value >1:
                raise Exception('MULTIPLE',' VALUES')
    
    def __diagonalAttacks(self): # mozliwe do poprawy !!!
        diagonalAttacks = 0
        columns = np.arange(0,self.size+1)
        for i,h1 in zip(columns[:-1],self.board[:-1]):
            for j,h2 in zip(columns[i+1:], self.board[i+1:]):
                if i != j and np.abs(i-j) ==np.abs(h1-h2):
                    diagonalAttacks += 1
        return diagonalAttacks
        
    def cutOutVector(self,begin,end ):
        return deepcopy(self.board[begin:end]) # włączamy indeks końca
    
    def injectVector(self,particle,begin,end):
        cutOut = deepcopy(self.board[begin:end])
        self.board[begin:end] = particle 
            
    def getValueOfIndex(self,index):
        return self.board[index]
    
    def setValueOfIndex(self,value,index):
        self.board[index] = value
    
    def getIndexOfValue(self,value,start,stop):
        board = list(self.board)
        return board.index(value,start,stop)
        
    def mutateAtIndexes(self,firstIndex,secondIndex):
        self.board[firstIndex], self.board[secondIndex] = self.board[secondIndex],self.board[firstIndex]
        self.CHECK()
        


# In[3]:


def printQueensPositions(chessboard: Chessboard):
    """
    wyświetl pozycje hetmanów na szachownicy w czytelny dla człowieka sposób
    """
    board = chessboard.getPositions()
    width = np.size(board) 
    print("{0}".format(board[:]+1)) # +1 dla czytelności 


# In[4]:


def printPopulation(population ):
    print("-------------------------------------------------------------------")
    print("  Subject | Subject code ")
    print("__________________")
    for i, chessboard in enumerate(population):
        print("  {}.  |".format(i+1),end=" ")
        printQueensPositions(chessboard)
    print("__________________")


# In[5]:


def printStats(result: Chessboard, BestScore,totalTime):
    """
    wyświetl statystyki jednej iteracji eksperymentu
    """
    
    printQueensPositions(result)
    print("Best Score: {0}".format(BestScore))
    print("Time spent: {0}".format(totalTime), end='\n\n')


# In[6]:


def drawPlots(size, bestData, avgScore):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,10))
    
                         
    
    ax1.plot(range(len(bestData)), bestData, marker='o')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title("{} Queens | Best Individual".format(size))
    ax1.set_xlabel("n Generation")
    ax1.set_ylabel("Best Score")

    
    ax2.plot(range(len(avgScore)),avgScore, marker='o')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title("{} Queens | Mean Scores".format(size))
    ax2.set_xlabel("n Generation")
    ax2.set_ylabel("Mean Scores")
    
    plt.show()


# In[7]:


class PMX:

    def __init__(self,X,Y,n):
        self.X = X
        self.Y = Y
        self.n =n
    
    def cross(self,begin,end):
        firstParticle, secondParticle  = self.__cutOutParticles( begin, end)
        
        self.X.injectVector(secondParticle,begin,end)
        self.Y.injectVector(firstParticle,begin,end)
        
        firstMap, secondMap = self.__createMappingDicts(begin,end)
        
        self.__correctOffspring(self.X,firstMap,begin, end )
        self.__correctOffspring(self.Y,secondMap,begin, end )
        return self.X, self.Y
        
    
    def __cutOutParticles(self, begin, end):
        firstParticle  = self.X.cutOutVector(begin, end)
        secondParticle = self.Y.cutOutVector(begin, end)
        return firstParticle, secondParticle
    
    def __createMappingDicts(self,begin,end):
        firstMap = {}
        secondMap ={}
        for i in range(begin,end):
            firstValue = self.X.getValueOfIndex(i)
            secondValue = self.Y.getValueOfIndex(i)
            if firstValue != secondValue:
                firstMap[firstValue] = secondValue
                secondMap[secondValue] = firstValue
            
        firstMap = self.__correctMapping(firstMap)
        secondMap = self.__correctMapping(secondMap)
        return firstMap,secondMap
    
    def __correctMapping(self,mapping):
        MainStack = []
        newMap ={}
        for key,value in mapping.items():
            stack =[]
            if [key,value] not in MainStack:
                stack.append([key,value])
                stack = self.__searchForKeyFromValue(mapping,stack)
                MainStack +=stack
                newKey,newValue = self.__getNewKeyAndValue(stack)
                newMap[newKey]=newValue
        return newMap
    
    def __searchForKeyFromValue(self,mapping,stack):
        while True:
            oldKey,oldValue = self.__popStack(stack)
            keys = list(mapping.keys())
            try:
                index = keys.index(oldValue)
                newKey =keys[index]
                newValue =mapping[newKey]
                if [newKey,newValue] not in stack:
                    stack.append([newKey,newValue])
                else:
                    return stack
            except ValueError:
                return stack
        
    
    def __popStack(self,stack):
        return stack[-1][0],stack[-1][1]
    
    def __getNewKeyAndValue(self,stack):
        return stack[0][0], stack[-1][-1]
    
    def __correctOffspring(self,chessboard,mappingList,begin, end): # TODO można poprawić
        for key, value in mappingList.items():
            try:
                index = chessboard.getIndexOfValue(key,0,begin)
                chessboard.setValueOfIndex(value,index)
            except ValueError:
                try:
                    index = chessboard.getIndexOfValue(key,end,self.n)
                    chessboard.setValueOfIndex(value,index)
                except ValueError:
                    pass


# In[8]:


class EvolutionAlgorithm:
    
    def __init__(self, chessboardSize=4, populationSize=5 ,crossDiscriminator=0.7,
                 mutationDiscriminator=0.2,maxGeneration=1000, FFMAX=0):
        
        self.__generation = 0
        self.__FFMAX = FFMAX
        self.__chessboardSize = chessboardSize
        self.__populationSize = populationSize
        self.__crossDiscriminator = crossDiscriminator
        self.__mutationDiscriminator = mutationDiscriminator
        self.__maxGeneration = maxGeneration
        self.__generateInitialPopulation()
        printPopulation(self.__initialPopulation)
        self.bestData =[]
        self.meanScore = []
        
    def __generateInitialPopulation(self):
        self.__initialPopulation = []
        for i in range(populationSize):
            rng = np.random.default_rng()
            boardState = rng.permutation(self.__chessboardSize )
            chessboard = Chessboard(n=self.__chessboardSize, board_state=boardState)
            self.__initialPopulation.append(chessboard)
            
    def doTheEvolution(self): #main function

        currentPopulation = self.__initialPopulation
        evaluateScroes = self.__evaluate(currentPopulation,self.__populationSize )
        BestIndex = self.__getIndexOfBestPopulation(evaluateScroes)
        self.__appendMeanScores(evaluateScroes)
        self.__appendBestData(evaluateScroes[BestIndex])
        
        while(self.__isGenerationUnderLimit() and self.__isSolutionNotFound(evaluateScroes,BestIndex)):
            self.__newPopulation = self.__selection(currentPopulation)
            self.__crossover()
            self.__mutation()
            evaluateScroes = self.__evaluate(self.__newPopulation,self.__populationSize )
            BestIndex = self.__getIndexOfBestPopulation(evaluateScroes)
            currentPopulation = self.__newPopulation
            self.__increseGenerationCounter()
            
            self.__appendMeanScores(evaluateScroes)
            self.__appendBestData(evaluateScroes[BestIndex])
            
        BestScore = evaluateScroes[BestIndex]
        return currentPopulation[BestIndex], BestScore ,self.bestData, self.meanScore
        
    
    def __evaluate(self,currentPopulation, size): # zwróć liczbę ataków między hetmanami na szachownicy 
        evaluateScroes = np.zeros(size,dtype=np.int8)
        currentPopulation = self.__checkIfTypeOfList(currentPopulation)        
        for i, chessboard in zip(range(size),currentPopulation):
            evaluateScroes[i] = chessboard.getNumberOfAttacks()
        return evaluateScroes
    
    def __checkIfTypeOfList(self,subject):
        if type(subject) != list:
            subject = [subject]
        return subject
    
    def __getIndexOfBestPopulation(self, evaluateScroes):
        return np.argmin(evaluateScroes)
    
    def __isGenerationUnderLimit(self):
        return self.__generation < self.__maxGeneration
    
    def __isSolutionNotFound(self,evaluateScroes,BestIndex):
        BestScore = evaluateScroes[BestIndex]
        return BestScore > self.__FFMAX
    
    def __selection(self,currentPopulation): # selekcja turniejowa
        newPopulation = []
        for i in range(populationSize):
            firstOpponentIndex,secondOpponentIndex = self.__drawIndexes(self.__populationSize)
            if firstOpponentIndex != secondOpponentIndex:
                firstOpponent = currentPopulation[firstOpponentIndex]
                secondOpponent = currentPopulation[secondOpponentIndex]
                winner = firstOpponent if self.__isFirstTheWinner(firstOpponent,secondOpponent ) else secondOpponent
                newPopulation.append(deepcopy(winner))
            else:
                withoutRival = currentPopulation[i]
                newPopulation.append(deepcopy(withoutRival))
        return newPopulation
            
    def __drawIndexes(self, size):
        first  = np.random.randint(low=0, high=size-1)
        second = np.random.randint(low=0, high=size-1)
        return first,second
    
    def __isFirstTheWinner(self,firstOpponent,secondOpponent ):
        firstScore  = self.__evaluate(firstOpponent,1)
        secondScore = self.__evaluate(secondOpponent,1)
        return firstScore <= secondScore
    
    def __crossover(self):
        for index in range(0,self.__populationSize-2,2):
            randomState = self.__getRandomValue()
            if randomState <= self.__crossDiscriminator:
                self.__cross(index, index +1)
            
    def __getRandomValue(self):
        return np.random.rand(1)[0]
    
    def __cross(self, firstIndex, secondIndex):
        firstChessboard                = self.__newPopulation[firstIndex]
        secondChessboard               = self.__newPopulation[secondIndex]
        begin, end                     = self.__setBoundaries()
        end +=1 #inclusive end of vector 
        
        pmx = PMX(firstChessboard,secondChessboard, self.__chessboardSize)
        firstChessboard,secondChessboard =pmx.cross(begin,end)
        firstChessboard.CHECK()
        secondChessboard.CHECK()
        self.__newPopulation[firstIndex] = firstChessboard
        self.__newPopulation[secondIndex] = secondChessboard
        
    def __setBoundaries(self):
        begin,end = self.__drawIndexes(self.__chessboardSize)
        while begin == end:
            begin,end = self.__drawIndexes(self.__chessboardSize)
        if end < begin:
            begin,end = end,begin
        return begin,end
     
    def __mutation(self):
        for index in range(self.__populationSize):           
            randomState = self.__getRandomValue()
            if randomState <= self.__mutationDiscriminator:
                self.__mutate(index)
                
    def __mutate(self,index):
        chessboard             = self.__newPopulation[index]
        firstIndex,secondIndex = self.__drawIndexes(self.__chessboardSize)
        while firstIndex == secondIndex:
            firstIndex,secondIndex = self.__drawIndexes(self.__chessboardSize)
        chessboard.mutateAtIndexes(firstIndex,secondIndex)
    
    def __increseGenerationCounter(self):
        self.__generation += 1
        
    def __appendBestData(self,best):
        self.bestData.append(best)
        
    def __appendMeanScores(self,evaluateScores):
        score = np.mean(evaluateScores)
        self.meanScore.append(score)


# In[9]:


#MAIN 
crossDiscriminator    = 0.75
mutationDiscriminator = 0.3
populationSize        = 20
boardStart            = 4
boardStop             = 100
MAXGENERATION         = 10000
boardSize             = np.arange(boardStart,boardStop+1)

for size in boardSize:
    print("................................................................................")
    print("Number of Queens: {}".format(size))
    print("Crossing Discriminator: {}".format(crossDiscriminator))
    print("Mutation Discriminator: {}".format(mutationDiscriminator))
    start             = time()
    evolution         = EvolutionAlgorithm(chessboardSize =size, populationSize=populationSize ,
                                            crossDiscriminator=crossDiscriminator,mutationDiscriminator=mutationDiscriminator,
                                           maxGeneration=MAXGENERATION)
    result, BestScore,bestData, avgScore = evolution.doTheEvolution()
    stop              = time()
    totalTime         = stop - start
    printStats(result, BestScore,totalTime )
    drawPlots(size, bestData, avgScore)


# In[ ]:




