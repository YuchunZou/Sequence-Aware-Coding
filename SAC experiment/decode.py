from mpi4py import MPI
import math
import numpy as np
import time
import sys

    # EpsilonMath choose the parity from the bottom first, because math's bottom's encoding complexity is lower
    # and the parity num from the bottom rows is always bettern than the top rows

    # one potential problem is the parity data sub-tasks


    # use as much as sub-tasks from the bottom rows as much as possible!!
     
    # 不需要分成两个
    # From the design result, EpsilonSearch choose the parity from the top first, 
    # because Search's top encoding complexity is lower
    # and the parity num from the bottom rows is always bettern than the top rows

    # for EpsilonSearch, we need to show that in series 3, EpsilonSearch is better than GLO and first
    # increase and then decrease

    # use as much as sub-tasks 最底下！！ rows as much as possible!! to show epsilon math is the best
    # 而且要保证最底下的一定个数比上一行的小！！ potential problem! 底下的比上面的多！！

def decode(ApX, G, helper, num_of_workers, p, t, delta, size, num_of_missing_original_subtasks, ENC):
    num_of_sub_tasks_for_decoding = set()
    decoding_matrix_multiplication_num = 0
    parity_num_for_decoding = 0
    result = []
    # data = np.random.rand(int(size[0] / (num_of_workers * p)), size[2])


    for i in range(p + t - 1, p - 1, -1):  # for every row from back to the front
        if len(helper[i]) > 0:  #need to decode
            # i - p is the index of parity rows
            # the way of decoding should be thought carefully
            if ENC == 'Epsilon':
                if num_of_missing_original_subtasks == 0:
                        break
                if len(helper[i]) > num_of_missing_original_subtasks:
                    helper[i] = helper[i][:num_of_missing_original_subtasks]
                    num_of_missing_original_subtasks = 0

                else:
                    num_of_missing_original_subtasks -= len(helper[i])


            parity_num_for_decoding += len(helper[i])

            num_of_sub_tasks_for_decoding.update(delta[i - p])
            
            # real decode
            decode_G = G[i - p][:len(delta[i-p]) * num_of_workers - len(helper[i])]
            helper[i].sort()
            for item in helper[i]:
                decode_G = np.r_[decode_G, [G[i - p][item]]] 

            # k = len(delta[i - p]) * num_of_workers
            # decode_G = np.identity(k)[:k - len(helper[i])]
            # v = np.vander(np.array([j/3 for j in range(1, k + 1)]), increasing=True)
            # decode_G = np.r_[decode_G, v[:len(helper[i])]] 

            # start_time = time.time()
            
            decode_G = np.linalg.pinv(decode_G)

            # print("decode_G time")
            # print(decode_G.shape)
            # end_time = time.time()
            # print(end_time - start_time)

            # start_time = time.time()

            ApX_decode = []
            for item in range(len(delta[i - p]) * num_of_workers - len(helper[i])):
                ApX_decode.append(np.ones((int(size[0] / (num_of_workers * p)), size[2])))
            for column in helper[i]:
                ApX_decode.append(ApX[i][column - 1])

            # print("IO time")
            # print(decode_G.shape)
            # end_time = time.time()
            # print(end_time - start_time)

            # matrix multiplication
            temp = np.zeros((int(size[0] / (num_of_workers * p)), size[2]))
            for row in range(len(decode_G)):
                for column in range(len(decode_G[row])):
                    if decode_G[row][column] != 0:
                        temp += decode_G[row][column] * ApX_decode[column]
                        # temp += decode_G[row][column] * data
                        # decode_G[row][column] * data
                        decoding_matrix_multiplication_num += 1
                result.append(temp)
            # print(result)
    
    return len(num_of_sub_tasks_for_decoding) * num_of_workers, decoding_matrix_multiplication_num, parity_num_for_decoding, result


def Cauchy(m, n):
    x = np.array(range(n + 1, n + m + 1))
    y = np.array(range(1, n + 1))
    x = x.reshape((-1, 1))
    diff_matrix = x - y
    cauchym = 1.0 / diff_matrix
    return cauchym


def decode_GLO(ApX, G, helper, num_of_workers, p, t, delta, size, parity, c3les_ApX):
    num_of_sub_tasks_for_decoding = set()
    decoding_matrix_multiplication_num = 0
    parity_num_for_decoding = parity
    result = []
    # data = np.random.rand(int(size[0] / (num_of_workers * p)), size[2])


    decode_G = np.identity(p * num_of_workers)
    # v = np.vander(np.array([j/3 for j in range(1, len(delta[i]) * num_of_workers + 1)]), increasing=True)
    v = Cauchy(p * num_of_workers, p * num_of_workers)
    decode_G = np.r_[decode_G[:p * num_of_workers - parity], v[:parity]] 
    decode_G = np.linalg.pinv(decode_G)

    temp = np.zeros((int(size[0] / (num_of_workers * p)), size[2]))
    for row in range(len(decode_G)):
        for column in range(len(decode_G[row])):
            if decode_G[row][column] != 0:
                temp += decode_G[row][column] * c3les_ApX[column]
                # temp += decode_G[row][column] * data
                # decode_G[row][column] * data
                decoding_matrix_multiplication_num += 1
        result.append(temp)


    return p * num_of_workers, decoding_matrix_multiplication_num, parity_num_for_decoding, result



def decode_spinner(ApX, helper, num_of_workers, size, parity_map, n, k):
    decoding_matrix_multiplication_num = 0
    parity_num_total = 0
    parity_num_for_decoding = 0
    result = []
    for i in range(n):
        origin_num_each_row = 0
        parity_data_index_each_row = []
        for j in range(len(helper[i])):
            if helper[i][j] in parity_map[i]:
                parity_num_total += 1
                parity_data_index_each_row.append(j)
            else:
                origin_num_each_row += 1

        if origin_num_each_row >= k:
            continue
        else:
            parity_num_each_row = k - origin_num_each_row
            parity_num_for_decoding += parity_num_each_row

            decode_G = np.identity(k)[:k - parity_num_each_row]
            v = np.vander(np.array([j/3 for j in range(1, k + 1)]), increasing=True)
            decode_G = np.r_[decode_G, v[-parity_num_each_row:]] 

            # start_time = time.time()
            decode_G = np.linalg.pinv(decode_G)


            # print("time")
            # print(decode_G.shape)
            # end_time = time.time()
            # print(end_time - start_time)

            ApX_decode = []
            for item in range(origin_num_each_row):
                ApX_decode.append(np.ones((int(size[0] / (n * k)), size[2])))
            for column in range(parity_num_each_row):
                ApX_decode.append(ApX[i][column])

            for row in range(len(decode_G)):
                temp = np.zeros((int(size[0] / (n * k)), size[2]))
                for column in range(len(decode_G[row])):
                    if decode_G[row][column] != 0:
                        temp += decode_G[row][column] * ApX_decode[column]     # use different data, 
                        decoding_matrix_multiplication_num += 1
                result.append(temp)

    return decoding_matrix_multiplication_num, parity_num_total, parity_num_for_decoding, result

def print_time(start):
    end = time.time()
    print(float(end - start))

            

            