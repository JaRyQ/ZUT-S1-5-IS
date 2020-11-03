#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from time import time
from queue import SimpleQueue as queue 
from collections import deque
from copy import deepcopy as dcopy
from collections import Counter


# In[156]:


class Chessboard:
    def __init__(self,n = 4): # inicjalizacja
        self.size=n
        self.board = np.zeros((self.size),dtype=np.int)
        self.queens_counter =0 
    def position(self,row,column): # ustawienie królowej na szachownicy
        self.board[row] = column
    def print_board(self,print_full_board=True): # wyświetlenie szachownicy
        if not print_full_board:
            return
        full_chessboard = np.zeros((self.size,self.size),dtype=np.int)
        for position, queen_position in zip(range(0,self.size),self.board) :
            full_chessboard [queen_position][position] = 1
        print( full_chessboard, end='\n\n')
    def get_positions(self):
        return  self.board[:]
    def default_position(self,queens_array):
        self.board[:]  = queens_array[:]
    def add_queen_counter(self):
        self.queens_counter +=1
    def get_queen_counter(self):
        return self.queens_counter
    def set_queens_counter(self,n_queens):
        self.queens_counter = n_queens


# In[157]:


def print_queens_positions(chessboard: Chessboard):
    """
    wyświetl pozycje hetmanów na szachownicy w czytelny dla człowieka sposób
    """
    print(" queens positions: {0}".format(chessboard.get_positions()+1)) # +1 dla czytelności 


# In[158]:


def print_stats(n_queens,generate_counter, state_check_counter,correct_counter,timer):
    """
    wyświetl statystyki jednej iteracji eksperymentu
    """
    print("N_queens: {0}".format(n_queens))
    print("Number of correct states: {0}".format(correct_counter))
    print("Number of generated states: {0}".format(generate_counter))
    print("Number of checked states: {0}".format(state_check_counter))
    print("Time spent: {0}".format(timer), end='\n\n')


# In[160]:


#test na zawarcie jednej klasy strategy
class Strategy:
    def __init__(self,size : int, Brute_Force_enabled = True, BFS_strategy=True,print_full_board=True):
        self.bf_enabled = Brute_Force_enabled
        self.BFS_strategy =BFS_strategy#true =>BFS | false => DFS
        self.state_check_counter = 0
        self.generate_counter =0 
        self.correct_counter =0
        self.size = size
        self.q = deque()
        self.print_full_board_bool = print_full_board
        
        

    def print_opening(self):
        print('############################################################')
        if self.BFS_strategy:
            print('Breadth First Search(BFS) STRATEGY')
        else:
            print('Depth First Search(DFS) STRATEGY')
        if self.bf_enabled:
            print("Brute Force")
        else:
            print('Inteligent approach')
    def append_queue(self, state, queen_row,column_position):
        state.position(queen_row, column_position)
        self.q.append((dcopy(state.get_positions()), queen_row+1))
        self.generate_counter +=1
    def generate_primary_state(self):
        board = Chessboard(self.size)
        self.q.append((board.get_positions(),0))

    def inteligent_position(self, board,current_checked_column): 
        n_queen = board.get_queen_counter()
        board_slice = board.get_positions()[:n_queen+1]
        board_slice[n_queen] = current_checked_column
        self.state_check_counter += 1
        #drugi warunek
        counter = Counter(board_slice) 
        for values in counter.values(): 
            if values > 1: 
                return False
        for current_queen in range(0, n_queen):
            if np.abs(current_queen-n_queen) == np.abs(board_slice[current_queen]-current_checked_column):
                return False
        return True
        
        
    def generate_succesor(self,state): #queen_row => czyli jakiego hetmana będziemy ustawiać
        
        queen_row = state.get_queen_counter()
        primary_state = state.get_positions()
        if self.bf_enabled : #prymitywne podejście
            """
            Zwróć wysztkie tablice z dodanym jednym hetmanem umieszczonym w dowolnym miejscu
            """

            for column_position in range(0, self.size):
                self.append_queue(state,queen_row,column_position)
                state.default_position(primary_state)
        else: # funkcja smart
            """
            zwróc tylko poprawne wektory (bez ataków)
            """
            
            for column_position in range(0, self.size):
                    if self.inteligent_position(state,column_position):
                        self.append_queue(state,queen_row,column_position)
                        state.default_position(primary_state)       
            #print(self.q, end='\n\n')
                    
                    
        
    def check_final_board_state(self, board_state): # test osiągniecia celu
        """
        test ośiągnięcia celu
        """
        if self.bf_enabled:
            if  board_state.get_queen_counter() < self.size: # jeżeli nie umieściliśmy wszystkich hetmanów zwacamy falsz
                return False
            self.state_check_counter += 1
            #drugi warunek 
            counter = Counter(board_state.get_positions()) 
            for values in counter.values(): 
                if values > 1: 
                    return False
            #trzeci warunek
            for i,h1 in zip(range(0, self.size), board_state.get_positions()):
                for j,h2 in zip(range(0, self.size), board_state.get_positions()):
                    if i != j and np.abs(i-j) ==np.abs(h1-h2)  :
                        return False
            self.correct_counter +=1 
            return True
        else:
            if board_state.get_queen_counter() ==self.size:
                self.correct_counter +=1
                return True
            
            return False
    
    def main_loop(self):
        while len(self.q): # dopóki lista stanów do przeszukania nie jest pusta
            if self.BFS_strategy: #STRATEGIA BFS
                queens_state, n_queens = self.q.popleft() # pobierz z początku listy najstarszy stan
            else: #STRATEGIA DFS
                queens_state, n_queens = self.q.pop() # pobierz z końca listy ostatnio dodany stan
            oldest_state = Chessboard(self.size)
            
            oldest_state.default_position(queens_state)
            oldest_state.set_queens_counter(n_queens)
            if self.check_final_board_state(oldest_state): # jeżeli został osiągnięty cel
                    print_queens_positions(oldest_state) # wyświetl pierwsze znalezione rozwiązanie
                    oldest_state.print_board(self.print_full_board_bool);
                    #break
            if n_queens != self.size:
                self.generate_succesor( oldest_state) # wygeneruj wszystkich potomków aktualnego stanu
        return self.generate_counter, self.state_check_counter, self.correct_counter 
    
    


# In[161]:


def Experiment(n_start=4, n_stop=20, print_full_board =1): #główna pętla eksperymentu
    if n_start <4:
        return 1# nie istnieją rozwiązania
    n_array = np.array(list(range(n_start,n_stop+1)))
    strategy_list = [True,False] #true =>BFS | false => DFS
    bf_enable_list =[True,False] #true => Brute Force | false =smart
    for n in n_array:# pętla po wielkości szachownicy 
        for strat in strategy_list:
            for bf_enabler in bf_enable_list:
                timer_start = time()
                strategy = Strategy(n,bf_enabler, strat,print_full_board)
                strategy.print_opening()
                strategy.generate_primary_state()
                generate_cnt, state_check_cnt, correct_cnt =strategy.main_loop()
                timer_stop = time()
                print_stats(n,generate_cnt, state_check_cnt,correct_cnt,timer_stop - timer_start)


# In[164]:


if Experiment(4,20,print_full_board =False):
    print("error n_start must be >= 4")


# In[ ]:




