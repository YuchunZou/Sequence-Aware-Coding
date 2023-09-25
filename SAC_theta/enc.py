import time
import sys
import numpy as np

def multiply(M, G):
    count = 0
    D = M[0].shape
    X = 1
    Y = D[-1]
    N, K = G.shape
    R = np.zeros((N, X, Y))
    for i in range(N):
        # print G[i]
        for j in range(K):
            if G[i, j] != 0:
                R[i] = R[i] + G[i, j] * M[j]
                count += 1
    return R

if __name__ == "__main__":
    code = sys.argv[1]
    D = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]
    times = int(sys.argv[5])
    n = int(sys.argv[6]) 
    u = int(sys.argv[7]) 
    c = int(sys.argv[8]) 
    ep = float(sys.argv[9])

    A = np.random.rand(n, int(D[0] / n * D[1]))
    B = np.random.rand(u, int(D[2] / u * D[1]))
 
    E = []
    if code == "static":
        from code1 import distribution
        from code1m import code1m 
        Dist = distribution(n, u, c)
        U = code1m(n, u, c, ep, Dist)
        GA = np.random.rand(c * n, n)
        GB = np.zeros((c * n, u))
        # print(U)
        for i in range(c):
            for j in range(n):
                for l in U[i]:
                    # print(i, j, i * n + j, l - 1)
                    GB[i * n + j, l - 1] = np.random.rand()
        for i in range(times):
            start = time.time()
            MA = multiply(A, GA)
            MB = multiply(B, GB)
            # from scipy.sparse import csr_matrix
            # MA = GA.dot(A)
            # GB0 = csr_matrix(GB)
            # MB = GB0.dot(B)
            end = time.time()
            E.append(end - start)
            print(E[-1])

    if code == "stochastic":
        from code1 import distribution
        from code7m import _code7m
        Dist = distribution(n, u, c)
        _, [CA, CB] = _code7m(n, u, c, ep, 100, Dist)
        GA = np.zeros((c * n, n))
        GB = np.zeros((c * n, u))
        for i in range(c):
            for j in range(n):
                for l in range(CB[i * n + j]):
                    GA[i * n + j, l] = np.random.rand()
                for l in range(CA[i * n + j]):
                    GB[i * n + j, l] = np.random.rand()
        for i in range(times):
            start = time.time()
            MA = multiply(A, GA)
            MB = multiply(B, GB)
            # from scipy.sparse import csr_matrix
            # GA0 = csr_matrix(GA)
            # MA = GA0.dot(A)
            # GB0 = csr_matrix(GB)
            # MB = GB0.dot(B)
            end = time.time()
            E.append(end - start)
            print(E[-1])

    print(np.mean(E))
