from mpi4py import MPI
import math
import numpy as np
import time
import sys

def encode_epsilon(A, p, t, delta, worldSize):
    L = A[0].shape[0]
    num_of_workers = len(A)
    G_of_t_rows = []
    decoding_matrix_multiplication_num = 0
    A_total_elements = A[0][:int(L / p)].shape[0] * A[0][:int(L / p)].shape[1]
    for i in range(len(delta)):
        # every delta should have a G
        # we just need the length of delta, don't need the value in delta
        G = np.identity(len(delta[i]) * num_of_workers)
        v = np.vander(np.array([j/3 for j in range(1, len(delta[i]) * num_of_workers + 1)]), increasing=True)
        G = np.r_[G, v[:num_of_workers]] 
        G_of_t_rows.append(G)
        for worker in range(num_of_workers): 
            temp = np.zeros((int(L / p), A[0].shape[1]))
            for column in range(len(delta[i]) * num_of_workers):
                temp += G[len(delta[i]) * num_of_workers + worker][column] * A[0][:int(L / p)]
                decoding_matrix_multiplication_num += 1
            A[worker] = np.vstack([A[worker], temp])
    print(decoding_matrix_multiplication_num * A_total_elements)
    return A, G_of_t_rows


def encode_glo(A, p, t, delta, worldSize):
    L = A[0].shape[0]
    num_of_workers = len(A)
    G_of_t_rows = []
    decoding_matrix_multiplication_num = 0
    A_total_elements = A[0][:int(L / p)].shape[0] * A[0][:int(L / p)].shape[1]
    for i in range(len(delta)):
        # every delta should have a G
        # we just need the length of delta, don't need the value in delta
        G = np.identity(len(delta[i]) * num_of_workers)
        v = np.vander(np.array([j/3 for j in range(1, len(delta[i]) * num_of_workers + 1)]), increasing=True)
        G = np.r_[G, v[:num_of_workers]] 
        G_of_t_rows.append(G)
        for worker in range(num_of_workers): 
            temp = np.zeros((int(L / p), A[0].shape[1]))
            for column in range(len(delta[i]) * num_of_workers):
                temp += G[len(delta[i]) * num_of_workers + worker][column] * A[0][:int(L / p)]
                decoding_matrix_multiplication_num += 1
            A[worker] = np.vstack([A[worker], temp])
    print(decoding_matrix_multiplication_num * A_total_elements)
    return A, G_of_t_rows


def Cauchy(m, n):
    x = np.array(range(n + 1, n + m + 1))
    y = np.array(range(1, n + 1))
    x = x.reshape((-1, 1))
    diff_matrix = x - y
    cauchym = 1.0 / diff_matrix
    return cauchym

def encode_c3les(A, p, t, delta, worldSize):
    L = A[0].shape[0]
    num_of_workers = len(A)
    G_of_t_rows = []
    decoding_matrix_multiplication_num = 0
    A_total_elements = A[0][:int(L / p)].shape[0] * A[0][:int(L / p)].shape[1]

    # same with glo, but use Cauchy
    for i in range(len(delta)):
        # every delta should have a G
        # we just need the length of delta, don't need the value in delta
        G = np.identity(len(delta[i]) * num_of_workers)
        # v = np.vander(np.array([j/3 for j in range(1, len(delta[i]) * num_of_workers + 1)]), increasing=True)
        v = Cauchy(len(delta[i]) * num_of_workers, len(delta[i]) * num_of_workers)
        G = np.r_[G, v[:num_of_workers]] 
        G_of_t_rows.append(G)
        for worker in range(num_of_workers): 
            temp = np.zeros((int(L / p), A[0].shape[1]))
            for column in range(len(delta[i]) * num_of_workers):
                temp += G[len(delta[i]) * num_of_workers + worker][column] * A[0][:int(L / p)]
                decoding_matrix_multiplication_num += 1
            A[worker] = np.vstack([A[worker], temp])
    print(decoding_matrix_multiplication_num * A_total_elements)
    return A, G_of_t_rows




def encode_MDS(A, size, n, k):
    r = n - k
    v = np.vander(np.array([j/3 for j in range(1, k + 1)]), increasing=True)
    G = v[:r]
    L = size[0]
    num_of_workers = n
    decoding_matrix_multiplication_num = 0
    A_total_elements = A[0].shape[0] * A[0].shape[1]

    for row in range(r): 
        temp = np.zeros((int(L / k), size[1]))
        for column in range(k):
            # print(temp)
            # print(G[row][column] * A[0])
            temp += G[row][column] * A[0]
            decoding_matrix_multiplication_num += 1
        A.append(temp)
    print(decoding_matrix_multiplication_num * A_total_elements)
    return A

def print_time(start):
    end = time.time()
    print(float(end - start))

# A = [np.ones((2, 2)), np.ones((2, 2))]
# p = 2
# t = 2
# delta = [[1,2], [1,2]]
# worldSize = 2
# print(encode_MDS(A, p, t, delta, worldSize))




