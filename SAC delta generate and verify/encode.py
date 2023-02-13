from mpi4py import MPI
import math
import numpy as np
import time
import sys

def encode(A, k, e, r, d, g, worldSize):
    L = A[0].shape[0]
    num_of_workers = len(A)
    G = np.zeros((num_of_workers * e, num_of_workers * e))
    for i in range(e):
        G[i * num_of_workers: (i + 1) * num_of_workers, i * num_of_workers: (i + 1) * num_of_workers] = np.vander(np.array([j for j in range(1, num_of_workers + 1)]), increasing=True)
    for i in range(r):
        for worker in range(num_of_workers):  # can be changed to replication and encode faster!!
            temp = np.zeros((L / k, A[0].shape[1]))
            for column in range(num_of_workers * e):
                temp += G[i * e + worker][column] * A[0][:L / k]   # only need to choose the right size, don't need to choose the right position
            A[worker] = np.vstack([A[worker], temp])   # the row should start from the bottem then go to the last epsilon row
    return A, G


