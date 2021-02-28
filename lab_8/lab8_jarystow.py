#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from time import time
from collections import Counter
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().magic('matplotlib inline')


# In[3]:


class Chessboard:
    def __init__(self,n=4,board_state=None): # inicjalizacja
        self.size = n # do w
        self.board = np.zeros((self.size),dtype=np.int) if board_state is None else board_state  
            
    def getPositions(self):
        return  self.board[:]      
        
    def getNumberOfAttacks(self):
        numberOfAttacks =0
        #numberOfAttacks = self.__horizontalAttacks()
        numberOfAttacks += self.__diagonalAttacks()
        #print(numberOfAttacks)
        return numberOfAttacks
    
    def CHECK(self): # test poprawności wektora
        counter       = Counter(self.board)
        for value in counter.values():
            if value >1:
                #printQueensPositions(self)
                raise Exception('MULTIPLE',' VALUES')

    
    def __horizontalAttacks(self):
        counter       = Counter(self.board) 
        counterValues = list(counter.values())
        for i,value in enumerate(counterValues):
            counterValues[i] = value-1
        return np.sum(counterValues)
    
    def __diagonalAttacks(self): # mozliwe do poprawy !!!
        diagonalAttacks = 0
        for i,h1 in enumerate(self.board):
                for j,h2 in enumerate( self.board):
                    if i != j and np.abs(i-j) ==np.abs(h1-h2)  :
                        diagonalAttacks += 1
                        break
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
        


# In[4]:


def printQueensPositions(chessboard: Chessboard):
    """
    wyświetl pozycje hetmanów na szachownicy w czytelny dla człowieka sposób
    """
    board = chessboard.getPositions()
    width = np.size(board)
    #print("{0:{width}}".format(board[:]+1, width=width)) # +1 dla czytelności 
    print("{0}".format(board[:]+1)) # +1 dla czytelności 


# In[5]:


def printPopulation(population ):
    print("-------------------------------------------------------------------")
    print("  Subject | Subject code ")
    print("__________________")
    for i, chessboard in enumerate(population):
        print("  {}.  |".format(i+1),end=" ")
        printQueensPositions(chessboard)
    print("__________________")


# In[6]:


def printStats(result: Chessboard, BestScore,totalTime):
    """
    wyświetl statystyki jednej iteracji eksperymentu
    """
    
    printQueensPositions(result)
    print("Best Score: {0}".format(BestScore))
    print("Time spent: {0}".format(totalTime), end='\n\n')


# In[ ]:


class PMX:

    def __init__(X,Y,begin,end)


# In[6]:


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
        
        while(self.__isGenerationUnderLimit() and self.__isSolutionNotFound(currentPopulation,BestIndex)):
            self.__newPopulation = self.__selection(currentPopulation)
            self.__crossover()
            self.__mutation()
            evaluateScroes = self.__evaluate(self.__newPopulation,self.__populationSize )
            BestIndex = self.__getIndexOfBestPopulation(evaluateScroes)
            currentPopulation = self.__newPopulation
            self.__increseGenerationCounter()
            
        BestScore = self.__evaluate(currentPopulation[BestIndex],1)
        return currentPopulation[BestIndex], BestScore
        
    
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
    
    def __isSolutionNotFound(self,currentPopulation,BestIndex):
        BestScore = self.__evaluate(currentPopulation[BestIndex],1)
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
        
        print("\nbefore", firstChessboard.board, secondChessboard.board)
        
        firstParticle, secondParticle  = self.__cutOutParticles(firstChessboard,secondChessboard, begin, end)
        print(begin,end)
        print("particles",firstParticle, secondParticle)
        
        firstChessboard.injectVector(secondParticle,begin,end)
        secondChessboard.injectVector(firstParticle,begin,end)
        print("after",firstChessboard.board, secondChessboard.board)
        
        firstMap, secondMap = self.__createMappingDicts(begin,end, firstChessboard, secondChessboard)
        
        self.__correctOffspring(firstChessboard,firstMap,begin, end )
        self.__correctOffspring(secondChessboard,secondMap,begin, end )
        print("corrected",firstChessboard.board, secondChessboard.board,end="\n")
        firstChessboard.CHECK()
        secondChessboard.CHECK()
        
    def __cutOutParticles(self,firstChessboard,secondChessboard, begin, end):
        firstParticle  = firstChessboard.cutOutVector(begin, end)
        secondParticle = secondChessboard.cutOutVector(begin, end)
        return firstParticle, secondParticle
    
    def __createMappingDicts(self,begin,end, firstChessboard, secondChessboard):
        firstMap = {}
        secondMap ={}
        for i in range(begin,end):
            firstValue = firstChessboard.getValueOfIndex(i)
            secondValue = secondChessboard.getValueOfIndex(i)
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
        oldKey,oldValue = self.__popStack(stack)
        keys = list(mapping.keys())
        try:
            index = keys.index(oldValue)
            newKey =keys[index]
            newValue =mapping[newKey]
            stack.append([newKey,newValue])
            return self.__searchForKeyFromValue(mapping,stack)
    
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
                    index = chessboard.getIndexOfValue(key,end,self.__chessboardSize)
                    chessboard.setValueOfIndex(value,index)
                except ValueError:
                    pass
        
        
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


# In[ ]:


#MAIN 
crossDiscriminator    = 0.7
mutationDiscriminator = 0.2
populationSize        = 10
boardStart            = 4
boardStop             = 20
MAXGENERATION         = 1000
boardSize             = np.arange(boardStart,boardStop+1)

for size in boardSize:
    print("................................................................................")
    print("Number of Queens: {}".format(size))
    start             = time()
    evolution         = EvolutionAlgorithm(chessboardSize =size, populationSize=populationSize ,
                                            crossDiscriminator=crossDiscriminator,mutationDiscriminator=mutationDiscriminator,
                                           maxGeneration=MAXGENERATION)
    result, BestScore = evolution.doTheEvolution()
    stop              = time()
    totalTime         = stop - start
    printStats(result, BestScore,totalTime )


# In[ ]:


#Nalezy wykonac wykres zmiennosci wartosci funkcji przystosowania
#najlepszego osobnika w generacjach oraz sredniej wartosci funkcji przystosowania z danej
#populacji tez w generacjach (os X - generacje, os Y - wartosc funkcji przystosowania).


# In[ ]:


#TEST czy algorytm działa
from itertools import cycle

def cut(X,begin,end):
    return deepcopy(X[begin:end]) 

def inject(X,begin,end,values):
    X[begin:end] =values
    return X

def mapping(X,Y,begin,end):    
    mapping = {}
    for i in range(begin,end):
        if X[i] != Y[i]:
            mapping[X[i]] = Y[i]
    return mapping


def getNewKeyAndValue(stack):
    return stack[0][0], stack[-1][-1]
def popStack(stack):
    return stack[-1][0],stack[-1][1]

def searchForKeyFromValue(mapping,stack):
    oldKey,oldValue = popStack(stack)
    #print("oldKey: ",oldKey)
    #print("oldValue: ", oldValue)
    keys = list(mapping.keys())
    try:
        index = keys.index(oldValue)
        newKey =keys[index]
        newValue =mapping[newKey]
        #print("newKey: ",newKey)
        #print("newValue: ", newValue)
        stack.append([newKey,newValue])
        return searchForKeyFromValue(mapping,stack)
    
    except ValueError:
        return stack
    

    
def correctMap(mapping):
    MainStack = []
    newMap ={}
    for key,value in mapping.items():
        stack =[]
        #print("Key in main Loop", key)
        #print("value in main Loop", value)
        if [key,value] not in MainStack:
            stack.append([key,value])
            stack = searchForKeyFromValue(mapping,stack)
            MainStack +=stack
            newKey,newValue = getNewKeyAndValue(stack)
            #print("set ",newKey," to ",newValue)
            newMap[newKey]=newValue
            #print("end recursion:", stack,end='\n\n')
            #deleteKeys(mapping, stack)
    return newMap
    #deleteKeys(mapping,routeStack)
    
def correctVector(X,mapping,begin,end):
    for key, value in mapping.items():
        try:
            print("index of key: " ,key, " -> ",X.index(key,0,begin))
            index = X.index(key,0,begin)
            X[index] = value
        except ValueError:
            try:
                print("index of key: " ,key, " -> ",X.index(key,end))
                index = X.index(key,end)
                X[index] = value
            except ValueError:
                pass
    return X
    
def MainTest(X,Y,begin,end):
    Xcut = cut(X,begin,end)
    Ycut = cut(Y,begin,end)
    #print("Xcut: ", Xcut)
    X = inject(X,begin,end,Ycut)
    Y = inject(Y,begin,end,Xcut)
    
    mappingX = mapping(X,Y,begin,end)
    mappingY = mapping(Y,X,begin,end)
    
    #print("mappingX: ",mappingX)
    mappingX = correctMap(mappingX)
    mappingY = correctMap(mappingY)
    
    #print("new mappingX: ",mappingX)
    return correctVector(X,mappingX,begin,end), correctVector(Y,mappingY,begin,end)



# In[ ]:


mapp={1:6,6:3, 4:5,3:2}
X = [3, 0, 7, 5, 4, 1, 2, 6]
Y = [7, 3, 5, 2, 4, 1, 6, 0]

#podmianka
print("X: ",X, "Y: ",Y)
#Xcut = [0, 7, 5, 4, 1, 2]
#Ycut = [3, 5, 2, 4, 1, 6]

begin = 1
end = 7
MainTest(X,Y,begin,end)


# In[ ]:





# In[ ]:




