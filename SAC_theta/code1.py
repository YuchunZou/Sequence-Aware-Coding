# third

from fig1 import occurance
from fig1 import strict_c_occurance
from fig1 import strict_u_occurance
from fig1 import strict_occurance

import numpy as np

def distribution(n, u, c):
    ret = {}
    for j in range(1, c + 1):
        total = strict_c_occurance(n, u, j)
        for i in range(1, u + 1):
            ret[(i, j)] = strict_c_occurance(n, i, j) / total
    return ret

def code1(n, u, c, ep, D):
    # D = distribution(n, u, c)
    ret = [0 for i in range(c)]
    for j in range(c):
        for i in range(u):
            if D[(i + 1,  j + 1)] >= ep:
                ret[j] = list(range(1, i + 2))
                break
    return ret

def generator1(n, u, c, ep, D):
    C = code1(n, u, c, ep, D)
    # print(C)
    G = np.zeros(((u + c) * n, u * n))
    for i in range(u * n):
        G[i, i] = 1
    for j in range(c):
        for m in range(n):
            coeffients = np.random.rand(len(C[j]) * n)
            t = 0
            for i in C[j]:
                for l in range(n):
                    G[u * n + j * n + m, (u - i) * n + l] = coeffients[t]
                    t += 1
    # print(np.count_nonzero(G))
    return G, np.mean([len(x) for x in C]) * n

def test_failure1(n, u, c, ep, D):
    G, _ = generator1(n, u, c, ep, D)

    P = [0 for i in range(n)]
    numbers = range(1, u + c + 1) 
    length = n
    target = u * n

    import itertools
    iterable = itertools.product(numbers, repeat = length)
    predicate = lambda x: (sum(x) == target)
    vals = filter(predicate, iterable)

    def pattern_to_rows(P):
        ret = []
        for i in range(n):
            for j in range(P[i]):
                ret.append(j * n + i)
        return ret

    ret = [0, 0]
    for P in vals:
        ind = pattern_to_rows(P)
        Gd = G[np.ix_(ind, list(range(0, u * n)))]
        # up = max(P)
        # lo = min(P) * (-1)
        # return np.linalg.matrix_rank(Gd) == x * n, sum([np.abs(x) for x in P]) / 2, max(P), min(P) * (-1)
        if np.linalg.matrix_rank(Gd) == u * n:
            ret[0] += 1
        ret[1] += 1
    return ret[0] / float(ret[1])

def test_failure_random1(n, u, c, ep, times, D):
    G, C = generator1(n, u, c, ep, D)

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
    
    def pattern_to_rows(P):
        ret = []
        for i in range(n):
            for j in range(u - P[i]):
                ret.append(j * n + i)
        return ret
    
    ret = [0, 0]
    for i in range(times):
        pattern(n, 0)
        ind = pattern_to_rows(P)
        Gd = G[np.ix_(ind, list(range(0, u * n)))]
        # up = max(P)
        # lo = min(P) * (-1)
        if np.linalg.matrix_rank(Gd) == u * n:
            ret[0] += 1
        ret[1] += 1
    return ret[0] / float(ret[1]), C

def test1():
    u = 4
    c = 3
    n = 5
    D = distribution(n, u, c)

    times = 10000

    ret = []
    step = 100
    ep = 0
    for i in range(0, step + 1):
        ep = i / step
        # ep = 0.1
        # test1 = test_failure1(n, u, c, ep, D)
        test2, C = test_failure_random1(n, u, c, ep, times, D)
        # if test2 < ep:
        #     print(ep, test2)
        ret.append([ep, test1, test2])
        ep += step
        # print(i, "/", step, "\t", test1, "\t", test2, "\t", C)
        print(i, "/", step, "\t", test2, "\t", C)
    # print(ret)
    return ret


if __name__ == "__main__":
    test1()
