import math
import numpy as np
import time
import sys
from sympy.utilities.iterables import partitions

def GenerateAllOriginalParityCombinations(epsilon, w, p, t):
    # https://docs.sympy.org/latest/modules/utilities/iterables.html
    # m: limits number of parts in partition
    # k: limits the numbers that are kept in the partition
    AllCombinations = []
    # i is the number of columns the original data can be in, 
    # the left threshold is minimum columns
    # the right threshold is to make epsilon parity data can be in
    NumOfOriginalWorkers = []
    for i in range(int(math.ceil(float(epsilon) / p)) , int(w - math.ceil(float(epsilon) / t)) + 1):
        # 
        for OriginalItemCombination in partitions(epsilon, m = p, k = i): 
            # Generate all original integer combinations
            OriginalKeys = OriginalItemCombination.keys()
            # Python 2.7
            # OriginalKeys.sort(reverse=True)
            # Python 3
            sorted(OriginalKeys, reverse=True)
            if list(OriginalKeys)[0] in NumOfOriginalWorkers:
                continue
            TempOriginal = np.zeros((p, 1))
            OriginalRowIndex = p - 1
            for key in OriginalKeys:
                for j in range(OriginalItemCombination[key]):
                    TempOriginal[OriginalRowIndex - j] = key
                OriginalRowIndex -= OriginalItemCombination[key]

            # Generate all parity integer combinations
            for ParityItemCombination in partitions(epsilon, m = t, k = w - list(OriginalKeys)[0]): 
                ParityKeys = ParityItemCombination.keys()
                # Python 2.7
                # ParityKeys.sort(reverse=True)
                # Python 3
                sorted(ParityKeys, reverse=True)
                TempParity = np.zeros((t, 1))
                ParityRowIndex = 0
                for key in ParityKeys:
                    for j in range(ParityItemCombination[key]):
                        TempParity[ParityRowIndex + j] = key
                    ParityRowIndex += ParityItemCombination[key]
                AllCombinations.append([TempOriginal, TempParity])
        NumOfOriginalWorkers.append(i)
    return AllCombinations

# epsilon = 4
# w = 8  #number of workers
# p = 4  #number of paritions
# t = 4  #parity rows
# # for OriginalItemCombination in partitions(epsilon, m = p, k = 2):
# #     print("Keys")
# #     print(OriginalItemCombination.keys())
# #     print("Values")
# #     print(OriginalItemCombination.values())
# print(GenerateAllOriginalParityCombinations(epsilon, w, p, t))



            

 

























