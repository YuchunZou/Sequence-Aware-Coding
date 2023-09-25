# first  third

from fig1 import occurance
from fig1 import strict_c_occurance
from fig1 import strict_u_occurance
from fig1 import strict_occurance

import numpy as np

from code1 import distribution

def _generator7m(n, u, c, ep, D):
    A = []
    B = []

    C = []
    G = np.zeros(((u + c) * n, u * n))
    for i in range(u * n):
        G[i, i] = 1

    N = np.ceil(u * n * ep)
    for j in range(c):
        for m in range(n):
            # N = np.random.binomial(x * n, ep)
            low = int(np.ceil(N / n))
            if low <= 1:
                p = np.random.rand()
            else:
                p = np.random.uniform(D[(low - 1, j + 1)], 1)
            for i in range(max(0, low - 1), u):
                if p < D[(i + 1, j + 1)]:
                    if D[(i + 1, j + 1)] == 1:
                        i = np.random.randint(i, u)
                    break
            placements = np.random.choice(n, size = int(np.ceil(N / (i + 1))), replace = False)
            coefficients1 = np.random.rand(i + 1)
            coefficients2 = np.random.rand(n)
            C.append((i + 1) * len(placements))
            A.append(i + 1)
            B.append(len(placements))
            for i0 in range(i + 1):
                for l in placements:
                    G[u * n + j * n + m, (u - 1 - i0) * n + l] = coefficients1[i0] * coefficients2[l] 
    # print(np.count_nonzero(G))
    return G, np.mean(C), [A, B]

def generator7m(n, u, c, ep, D):
    G, C, _ = _generator7m(n, u, c, ep, D)
    return G, C

def test_failure_random7m(G, n, u, c, ep, times):
    P = [0 for i in range(n)]
    def pattern(n, sum):
        if n == 1:
            if sum <= u and sum >= -1 * c:
                P[0] = sum
                return True
            else:
                return False
        if sum <= u * n and sum >= -1 * c * n:
            while True: 
                P[n - 1] = np.random.randint(-1 * c, u + 1)
                if pattern(n - 1, sum - P[n - 1]) == True:
                    break
            return True
        else:
            return False
    # def pattern(n):
    #     S = set(range(0, n))
    #     sum = u * n
    #     for i in range(n):
    #         P[i] = 0
    #     count = 0
    #     while (count < sum):
    #         i = list(S)[np.random.randint(0, len(S))]
    #         a = np.random.randint(1, min(u + c - P[i], sum - count) + 1)
    #         P[i] += a
    #         count += a
    #         if P[i] == u + c:
    #             S.remove(i)
    #     # print(P)
    #     for i in range(n):
    #         P[i] = u - P[i]
    
    def pattern_to_rows(P):
        ret = []
        for i in range(n):
            for j in range(u - P[i]):
                ret.append(j * n + i)
        return ret
    
    ret = [0, 0]
    for i in range(times):
        pattern(n, 0)
        # pattern(n)
        ind = pattern_to_rows(P)
        Gd = G[np.ix_(ind, list(range(0, u * n)))]
        # up = max(P)
        # lo = min(P) * (-1)
        # import scipy.linalg as la
        # _, _, U = la.lu(Gd)
        # if np.abs(np.prod(np.diag(U))) > 1e-6:
        if np.linalg.matrix_rank(Gd) == u * n:
            ret[0] += 1
        ret[1] += 1
    return ret[0] / float(ret[1])

def _code7m(n, u, c, ep, times, D):
    ret = 10
    G = None
    C = None
    A = None
    B = None
    diff = 1
    for i in range(times):
        # print(i)
        G0, C0, [A0, B0] = _generator7m(n, u, c, ep, D)
        test = test_failure_random7m(G0, n, u, c, ep, 100) 
        if np.abs(test - ep) < diff:
            diff = np.abs(test - ep)
            ret = test
            G = G0
            C = C0
            A = A0
            B = B0
    return [ret, G, C], [A, B]

def code7m(n, u, c, ep, times, D):
    out, _ = _code7m(n, u, c, ep, times, D)
    return out


def test7m():
    u = 10
    c = 4
    n = 10 

    # ep = .2
    times = 100

    D = distribution(n, u, c)

    ret = []
    step = 4
    ep = 0
    for i in range(1, step + 1):
        ep = i / step
        # test1 = test_failure7m(n, u, c, ep, times)
        r, G, C = code7m(n, u, c, ep, times, D)
        # test2 = test_failure_random7m(G, n, u, c, ep, 10000)
        # if test < ep:
        #     print(ep, test)
        # ret.append([ep, r, test2])
        ep += step
        # print(i, "/", step, "\t", r, "\t", test2, "\t", C)
        print(i, "/", step, "\t", r, "\t", C)
    # print(ret)
    return ret

if __name__ == "__main__":
    test7m()
