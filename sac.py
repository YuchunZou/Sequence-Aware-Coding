from mpi4py import MPI
import sys
import numpy as np
import time

comm = MPI.COMM_WORLD
worldSize = comm.Get_size()
rank = comm.Get_rank()
processorName = MPI.Get_processor_name()

TaskMaster = worldSize - 1
n = TaskMaster
u = int(sys.argv[1])
c = int(sys.argv[2])
ep = float(sys.argv[3])
D = [int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])]
ENC = sys.argv[7]
delay = float(sys.argv[8])

# n = 5
# u = 4
# c = 3
# ep = 1
# D = [n, 1, u]
# ENC = "stochastic"
# delay = 0

import logging

if '-v' in sys.argv:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

logging.debug("[%d] Process %s started." % (rank, processorName))

def worker(rank, u, c, A, B, TaskMaster):
    sleep_time = np.random.exponential(delay)
    for i in range(1, u + 1):
        time.sleep(sleep_time)
        logging.debug("[%d] Process %d calculating uncoded subtask %d." % (rank, rank, i))
        Y = np.dot(A[i], B[i])
        request = comm.Ssend(Y, dest = TaskMaster, tag = i)
    for i in range(1, c + 1):
        time.sleep(sleep_time)
        logging.debug("[%d] Process %d calculating coded subtask %d." % (rank, rank, i))
        Y = np.dot(A[u + i - 1], B[u + i - 1])
        request = comm.Ssend(Y, dest = TaskMaster, tag = u + i)

def master(n, u, c, D, G):
    t_start = MPI.Wtime()
    M = []
    for i in range(0, n):
        M.append([])
        for j in range(u + c):
            M[i].append(np.zeros((int(D[0] / n), int(D[2] / u))))
    index = []
    loc = []
    num = 0
    parity = 0

    def print_time():
        time = MPI.Wtime()
        print(time - t_start)

    while(True):
        st = MPI.Status()
        comm.Probe(status=st)
        i = st.source
        j = st.tag
        comm.Recv([M[i][j - 1], MPI.FLOAT], source = i, tag = st.tag)
        index.append((j - 1) * n + i)
        loc.append((i, j - 1))
        logging.debug("[%d] worker %d partition %d received" % (rank, i, j))
        logging.debug("[%d] index received: [%s]" % (rank, " ".join(map(lambda x: str(x), index))))
        num += 1
        if j > u:
            parity += 1

        if (len(index) >= u * n):
            # print(index)
            Gd = G[np.ix_(index, list(range(0, u * n)))]
            if np.linalg.matrix_rank(Gd) == u * n:
                break

    t_comp = MPI.Wtime()
    logging.debug("[%d] Parity: %d" % (rank, parity))
    logging.debug("[%d] Total: %d" % (rank, num))
    logging.debug("[%d] Enough results obtained." % (rank))
    logging.debug("[%d] Time: %5.4fs" % (rank, t_comp - t_start))

    Me = np.zeros((len(index), int(D[0] * D[2] / u / n)))
    for x in range(len(index)):
        i, j = loc[x]
        Me[x] = M[i][j].reshape(1, int(D[0] * D[2] / u / n))
    # from scipy.sparse import csr_matrix
    # G0 = csr_matrix(G[np.ix_(index, list(range(0, u * n)))])
    # from scipy.sparse.linalg import spsolve
    # x = spsolve(G0, Me)

    G0 = np.linalg.pinv(G[np.ix_(index, list(range(0, u * n)))])
    # print(np.count_nonzero(G0))
    from enc import multiply
    x = multiply(Me, G0)
    t_dec = MPI.Wtime()
    logging.debug("[%d] Time: %5.4fs" % (rank, t_dec - t_comp))
    logging.info("%s", [t_comp - t_start, t_dec - t_comp, parity, num])
    
    
if rank != TaskMaster:
    logging.debug("[%d] Running from processor %s, rank %d out of %d processors." % (rank, processorName, rank, worldSize))
    t_start = MPI.Wtime()
    Ai = []
    Bi = []
    for i in range(u + c):
        Ai.append(np.zeros((int(D[0] / n), D[1])))
        Bi.append(np.zeros((D[1], int(D[2] / u))))
    logging.debug("[%d] %d loaded" % (rank, rank))
    comm.Barrier()
    worker(rank, u, c, Ai, Bi, TaskMaster)
    exit(0)

if rank == TaskMaster:
    def code(ENC):
        if ENC == "static":
            from code1 import distribution
            from code1m import generator1m 
            Dist = distribution(n, u, c)
            G, _ = generator1m(n, u, c, ep, Dist)
            return G
        if ENC == "stochastic":
            from code1 import distribution
            from code7m import code7m
            Dist = distribution(n, u, c)
            times = 100
            _, G, _ = code7m(n, u, c, ep, times, Dist)
            return G

    G = code(ENC)
    comm.Barrier()
    if G is not None:
        master(n, u, c, D, G)
    comm.Abort(0)
    exit(0)
