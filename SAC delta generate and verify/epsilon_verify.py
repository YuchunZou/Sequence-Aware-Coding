import math
import numpy as np
import time
import sys
import copy
from epsilon_generate_deltas_trial3  import *
from epsilon_generate_all_combinations import * 
from sympy.utilities.iterables import partitions
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from encode import *
from decode import *
def print_time(start):
    end = time.time()
    print(float(end - start))

w = 16  #number of workers
p = 15  #number of paritions
t = 2  #parity rows
epsilon = 15

# epsilon = 4
# w = 5  #number of workers
# p = 3  #number of paritions
# t = 3  #parity rows

start = time.time()
deltas = generate_final_deltas(epsilon, w, p, t)
print_time(start)
print(deltas)
# deltas = [[0, 1, 2, 3, 4, 5], [4, 5]]
AllCombinations = GenerateAllOriginalParityCombinations(epsilon, w, p, t)

if epsilon > t * (w - math.ceil(float(epsilon) / p)):
    print("Can't handle this case, epsilon is too large")
    exit()

# a = [[0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1],[0, 0, 1, 1, 1, 1, 1, 1],[0, 0, 1, 1, 1, 1, 1, 1],[0, 0, 1, 1, 1, 1, 1, 1],[0, 0, 1, 1, 1, 1, 1, 1],[1, 1, 0, 0, 1, 1, 1, 1],[0, 0, 1, 1, 0, 0, 1, 1],]
# a = [[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 1]]

def epsilon_verify(ListOriginal, ListParity, deltas):
    # Generate the adjacency matrix 
    AdjacencyMatrix = []
    # This uses the scipy Maximum Bipartite Matching API
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_bipartite_matching.html
    # We just need to generate the adjacency matrix and see if -1 is in the result
    # -1 means we can't find any parity data to cover this missing original data item

    # comment!!!
    for row in range(len(deltas)):
        DeltaInEachRow = deltas[row]  # 1, 2
        temp = np.zeros(epsilon)
        for item in DeltaInEachRow:  #item 1 , 2
            EpsilonNum = 0
            for i in range(item):
                EpsilonNum += int(ListOriginal[i][0])
            for j in range(EpsilonNum, EpsilonNum + int(ListOriginal[item][0])):
                temp[j] = 1
        for m in range(int(ListParity[row][0])):
            AdjacencyMatrix.append(temp)
    graph = csr_matrix(AdjacencyMatrix)
    if -1 in maximum_bipartite_matching(graph, perm_type='column'):
        return False

for combination in AllCombinations:
    ListOriginal = copy.deepcopy(combination[0])
    ListParity = copy.deepcopy(combination[1])
    # print("Original and parity")
    L1 = copy.deepcopy(ListOriginal)
    L2 = copy.deepcopy(ListParity)
    if epsilon_verify(ListOriginal, ListParity, deltas) == False:
        print("Deltas are ")
        print(deltas)
        print(L1)
        print(L2)
        print("Deltas is wrong.")
print("Deltas are ")
print(deltas)
print("Deltas is correct.")

































