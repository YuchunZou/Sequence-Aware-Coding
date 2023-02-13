import math
import numpy as np
import time
import sys
import copy
from sympy.utilities.iterables import partitions

# epsilon = 10
# w = 8  #number of workers
# p = 5  #number of paritions
# t = 4  #parity rows
# if epsilon > t * (w - math.ceil(float(epsilon) / p)):
#     print("Can't handle this case, epsilon is too large")
#     exit()

def generate_deltas(ListOriginal, ListParity, epsilon, w, p, t):
    pointer = 0
    while(ListOriginal[pointer] == 0):
        pointer += 1
    # original_dic = []
    # for i in range(p):
    #     original_dic.append([])
    deltas = []
    for i in range(t):
        deltas.append([])

    while(pointer < p):
        # First delete all the elements that's already been covered
        # number_of_original_data_already_been_covered = 0
        # for i in original_dic[pointer]:    # the parity rows that covering this original data rows
        #     number_of_original_data_already_been_covered += ListParity[i]

        # if ListOriginal[pointer] > number_of_original_data_already_been_covered:
        #     ListOriginal[pointer] -= number_of_original_data_already_been_covered
        #     for i in original_dic[pointer]:
        #         ListParity[i] = 0

        # Add elements to deltas and original_dic, also sort the original_dic to acheive descending order
        number_of_original_data_needed_to_cover = ListOriginal[pointer] 
        rows_with_parity_data = []
        for i in range(t - 1, -1, -1):
            if ListParity[i] != 0:
                rows_with_parity_data.append(i)

        for i in rows_with_parity_data:
            if number_of_original_data_needed_to_cover > ListParity[i]:
                number_of_original_data_needed_to_cover -= ListParity[i]
                ListParity[i] = 0
                # original_dic[pointer].append(i)
                deltas[i].append(pointer)

            else:
                ListParity[i] -= number_of_original_data_needed_to_cover
                number_of_original_data_needed_to_cover = 0
                # original_dic[pointer].append(i)
                deltas[i].append(pointer)
                pointer += 1
                break


            # elif number_of_original_data_needed_to_cover == ListParity[i]:
            #     number_of_original_data_needed_to_cover = 0 
            #     ListParity[i] = 0 
            #     original_dic[pointer].append(i)
            #     deltas[i].append(pointer)
            #     pointer += 1
            #     break
            # else:
            #     ListParity[i] -= number_of_original_data_needed_to_cover
            #     number_of_original_data_needed_to_cover = 0
            #     original_dic[pointer].append(i)
            #     deltas[i].append(pointer)
            #     pointer += 1
            #     break
            
        # elif ListOriginal[pointer] == number_of_original_data_already_been_covered:
        #     ListOriginal[pointer] = 0
        #     for i in original_dic[pointer]:
        #         ListParity[i] = 0
        #     pointer += 1

        # else: 
        #     number_of_original_data_needed_to_cover = ListOriginal[pointer]
        #     ListOriginal[pointer] = 0
        #     for i in original_dic[pointer]:
        #         if ListParity[i] >= number_of_original_data_needed_to_cover:
        #             ListParity[i] -= number_of_original_data_needed_to_cover
        #             number_of_original_data_needed_to_cover = 0
        #             break
        #         else:
        #             number_of_original_data_needed_to_cover -= ListParity[i]
        #             ListParity[i] = 0
        #     pointer += 1
    return deltas

def generate_final_deltas(epsilon, w, p, t):
    AllCombinations = []
    for OriginalItemCombination in partitions(epsilon, m = p, k = int(math.ceil(float(epsilon) / p))): 
        # Generate original data combinations
        OriginalKeys = OriginalItemCombination.keys()
        # Python 2.7
        # OriginalKeys.sort(reverse=True)
        # Python 3
        sorted(OriginalKeys, reverse=True)
        TempOriginal = np.zeros((p, 1))
        OriginalRowIndex = p - 1
        for key in OriginalKeys:
            for j in range(OriginalItemCombination[key]):
                TempOriginal[OriginalRowIndex - j] = key
            OriginalRowIndex -= OriginalItemCombination[key]
        # print(TempOriginal)
        # Generate all parity data combinations
        for i in range(int(math.ceil(float(epsilon) / t)) , int(w - math.ceil(float(epsilon) / p)) + 1):
            for ParityItemCombination in partitions(epsilon, m = t, k = i): 
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

    for ParityItemCombination in partitions(epsilon, m = t, k = int(math.ceil(float(epsilon) / t))): 
        # Generate parity data combinations
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

        
        # Generate all original data combinations
        for i in range(int(math.ceil(float(epsilon) / p)) , int(w - math.ceil(float(epsilon) / t)) + 1):
            for OriginalItemCombination in partitions(epsilon, m = p, k = i): 
                OriginalKeys = OriginalItemCombination.keys()
                # Python 2.7
                # OriginalKeys.sort(reverse=True)
                # Python 3
                sorted(OriginalKeys, reverse=True)
                TempOriginal = np.zeros((p, 1))
                OriginalRowIndex = p - 1
                for key in OriginalKeys:
                    for j in range(OriginalItemCombination[key]):
                        TempOriginal[OriginalRowIndex - j] = key
                    OriginalRowIndex -= OriginalItemCombination[key]
                AllCombinations.append([TempOriginal, TempParity])
    # print(AllCombinations[0])
    DeltaList = []
    for Combination in AllCombinations:
        OriginalData = copy.deepcopy(Combination[0])
        ParityData = copy.deepcopy(Combination[1])
        DeltaList.append(generate_deltas(OriginalData, ParityData, epsilon, w, p, t))

    FinalDelta = []
    for i in range(t):
        temp = []
        for j in range(len(DeltaList)):
            temp += DeltaList[j][i]
        FinalDelta.append(list(set(temp)))
    # print(FinalDelta)
    return FinalDelta

    
# generate_final_deltas(epsilon, w, p, t)















