from mpi4py import MPI
import math
import numpy as np
import time
import sys

def decode(ApB, G, helper, num_of_workers, k, e, r, d, g):
    for i in range(r):  # for every row
        if len(helper[k - 1 + k - (k + i)]) <  num_of_workers:  # if there are data needed to be decoded
            helper[k - 1 + k - (k + i)].sort()
            helper[k + i].sort()
            data = []
            decode_G = np.zeros((num_of_workers, num_of_workers))
            identity = np.identity(num_of_workers)
            vande = np.vander(np.array([j for j in range(1, num_of_workers + 1)]), increasing=True)

            for j in range(len(helper[k - 1 + k - (k + i)])):
                decode_G[j] = identity[helper[k - 1 + k - (k + i)][j]]
                data.append(ApB[k - 1 + k - (k + i)][helper[k - 1 + k - (k + i)][j]])
            for p in range(num_of_workers - len(helper[k - 1 + k - (k + i)])):
                decode_G[len(helper[k - 1 + k - (k + i)]) + p] = vande[helper[k + i][p]]
                data.append(ApB[k + i][helper[k + i][p]])
            
            data = np.reshape(data, (num_of_workers, -1))
            decode_G = np.linalg.pinv(decode_G)
            result = np.dot(decode_G, data)
            