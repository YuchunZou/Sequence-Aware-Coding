from mpi4py import MPI
import math
import numpy as np
import time
import sys
from encode import *
from decode import *

comm = MPI.COMM_WORLD
worldSize = comm.Get_size()
rank = comm.Get_rank()
processorName = MPI.Get_processor_name()
TaskMaster = 0

p = int(sys.argv[1])   #number of partitions
t = int(sys.argv[2])   #t parity rows
e = int(sys.argv[3])
size = [int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])]
num_of_workers = worldSize - 1
X = np.ones((size[1], size[2]))
ENC = sys.argv[7]

delta = [[9, 10, 11, 12, 13, 14, 15], [9, 10, 11, 12, 13, 14, 15]]
if ENC == 'GLO' or ENC == 'c3les':
    delta = []
    for i in range(t):
        delta.append([i for i in range(1, p + 1)])
# print(delta)
#delta = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12]]
#delta = [[1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12,  13], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13]]   # this should be an input, input the index of row + 1

if ENC != 'GLO' and ENC != 'Epsilon' and ENC != 'c3les':
    print("Encoding has to be GLO or Epsilon or c3les.")
    exit()
if size[0] % num_of_workers != 0:
    print("A's rows must be a multiple of n.")
    exit()
elif size[0] % (num_of_workers * p)  != 0:
    print("size[0] must be a multiple of num_of_workers * k.")
    exit()   
# elif e < t:
#     print("epsilon can't be less than r.")
#     exit()



if rank != TaskMaster:
    Ap = comm.recv(source=TaskMaster, tag=10000)
    comm.Barrier()
    data = np.random.exponential(0.0, size= num_of_workers)
    time.sleep(data[rank - 1])
    L = int(size[0] / (num_of_workers * p)) 
    for i in range(0, p + t):
        ApX = np.dot(Ap[i * L : (i + 1) * L], X)
        request = comm.Ssend(ApX, dest = TaskMaster, tag= i)
    exit()

if rank == TaskMaster:
    Data = []
    ApX = []
    ApXDecode = []
    A = []
    for i in range(p + t):
        ApX.append([])
        for j in range(num_of_workers):
            ApX[i].append(np.zeros((int(size[0] / (num_of_workers * p)), size[2])))
    for i in range(num_of_workers):
        A.append(np.ones((int(size[0] / num_of_workers),  size[1]))) 
    start = time.time()

    if ENC == 'Epsilon':
        Ap, G = encode_epsilon(A, p, t, delta, worldSize)
    if ENC == 'GLO':
        Ap, G = encode_glo(A, p, t, delta, worldSize)
    if ENC == 'c3les':
        Ap, G = encode_c3les(A, p, t, delta, worldSize)
    print_time(start)
    # print(G)
    start = time.time()
    for worker_num  in range(1, worldSize):
        comm.send(Ap[worker_num - 1], dest = worker_num , tag=10000)
    comm.Barrier()
    helper = []
    for i in range(p + t):
        helper.append([])
    total_received_subtasks = 0
    parity = 0
    total_num_of_original_subtasks = p * num_of_workers
    num_of_missing_original_subtasks = 0

    GLO_ApX = []


    if ENC == 'GLO' or ENC == 'c3les':
        num_of_received_original_subtasks = 0
        while(total_received_subtasks < total_num_of_original_subtasks):  # and number of parity should be less or equal to epsilon
            st = MPI.Status()
            comm.Probe(status=st)
            i = st.source
            j = st.tag
            comm.Recv(ApX[j][i - 1], source = i, tag = st.tag)
            total_received_subtasks += 1
            helper[j].append(i)   # start from worker 0, but tag will from worker 1
            if ENC == 'c3les':
                if j >= p:
                    parity += 1
            if ENC == 'GLO':
                if i in list(range(num_of_workers - math.ceil(t * num_of_workers / (p +t)) + 1, num_of_workers + 1)) and parity < t * num_of_workers:
                    parity += 1
                    GLO_ApX.append(ApX[j][i - 1])
                else: 
                    num_of_received_original_subtasks += 1
                    GLO_ApX.append(ApX[j][i - 1])
        if ENC == 'c3les':
            num_of_missing_original_subtasks = parity
        else: 
            num_of_missing_original_subtasks = (num_of_workers - math.ceil(t * num_of_workers / (p +t))) * num_of_workers - num_of_received_original_subtasks


    if ENC == 'Epsilon':
        num_of_received_original_subtasks = 0
        num_of_original_subtasks_in_must_complete_rows = 0
        rows_in_delta = []
        for i in range(len(delta)):
            rows_in_delta += delta[i]
        temp = list(set(rows_in_delta))
        rows_must_complete = [x for x in [i for i in range(1, p + 1)] if x not in temp]

        # when e >= 16, 
        # print("Hello")
        while(1):  # and number of parity should be less or equal to epsilon
            st = MPI.Status()
            comm.Probe(status=st)
            i = st.source
            j = st.tag
            comm.Recv(ApX[j][i - 1], source = i, tag = st.tag)
            total_received_subtasks += 1
            helper[j].append(i)   # start from worker 0, but tag will from worker 1
            if j < p:  # original subtasks:
                num_of_received_original_subtasks += 1
                if j in rows_must_complete:
                    num_of_original_subtasks_in_must_complete_rows += 1
            else:
                parity += 1
                if parity > e + 3:
                    print(str(parity) + " parity data number larger than epsilon + 3")
                    exit()

            num_of_missing_original_subtasks = total_num_of_original_subtasks - num_of_received_original_subtasks
            
            if num_of_missing_original_subtasks <= e  \
            and num_of_original_subtasks_in_must_complete_rows >= len(rows_must_complete) * num_of_workers \
            and parity >= num_of_missing_original_subtasks:
                # print(num_of_missing_original_subtasks)
                break

    print_time(start)

    start = time.time()
    #print(helper)
    result = []

    if ENC == 'GLO':
        num_of_sub_tasks_for_decoding, decoding_matrix_multiplication_num, parity_num_for_decoding, result \
        = decode_GLO(ApX, G, helper, num_of_workers, p, t, delta, size, parity, GLO_ApX)
    else: 
        num_of_sub_tasks_for_decoding, decoding_matrix_multiplication_num, parity_num_for_decoding, result \
        = decode(ApX, G, helper, num_of_workers, p, t, delta, size, num_of_missing_original_subtasks, ENC)
    print_time(start)
    one_piece = size[0] / (num_of_workers * p) * size[2]

    print(total_received_subtasks)
    print(parity)
    print(num_of_sub_tasks_for_decoding)
    print(decoding_matrix_multiplication_num * one_piece)
    print(parity_num_for_decoding)
    exit()



