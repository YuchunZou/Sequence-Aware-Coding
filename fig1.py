import numpy as np

def occurance(n, u, c):
    ret = 0
    total = u * n
    import math
    # for i in range(0, math.ceil((y * n) / (x + y + 1))):
    for i in range(0, n):
        # print(ret)
        if total >= 0:
            ret += math.comb(total + n - 1, n - 1) * math.comb(n, i) * math.pow(-1, i)
            total -= (c + u + 1)
        else:
            break
    return ret

# print(occurance(1, 1, 1, 1))

def strict_c_occurance(n, u, c):
    if c == 1:
        return occurance(n, u, c)
    else:
        return occurance(n, u, c) - occurance(n, u, c - 1)

def strict_u_occurance(n, u, c):
    if u == 1:
        return occurance(n, u, c)
    else:
        return occurance(n, u, c) - occurance(n, u - 1, c)

def strict_occurance(n, u, c):
    if u == 1:
        return strict_c_occurance(n, u, c)
    else:
        return strict_c_occurance(n, u, c) - strict_c_occurance(n, u - 1, c)

if __name__ == "__main__":

    N1 = [strict_c_occurance(5, x, 1) for x in range(1, 5)]
    N2 = [strict_c_occurance(5, x, 2) for x in range(1, 5)]
    N3 = [strict_c_occurance(5, x, 3) for x in range(1, 5)]

    print(N1, N2, N3)

    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    plt.style.use('default')
    font = {'family' : 'Helvetica', 'weight' : 'normal', 'size'   : 16, }
    lines = {'linewidth' : 2, 'color' : 'black'}
    axes = {'edgecolor' : 'black', 'grid' : False, 'titlesize' : 'medium'}
    grid = {'alpha' : 0.1, 'color' : 'black'}
    figure = {'titlesize' : 32, 'autolayout' : True, 'figsize' : "4, 12"}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', **lines)
    matplotlib.rc('axes', **axes)
    matplotlib.rc('grid', **grid)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(9, 3)

    ax1.set_ylabel(r'$N_c(n=5,u,c)$')
    # ax1.set_xlabel('$w$')
    ax1.bar(np.arange(1,5), N1, .9)
    # ax1.grid(True)
    plt.sca(ax1)
    plt.xticks(np.arange(1, 5), ["u=1", "u=2", "u=3", "u=4"])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.set_title("(a) c=1", y = 1.0)
    ax1.set_ylim([0, N1[-1]])
    ax1.grid("on")

    # ax2.set_ylabel('# placements')
    # ax1.set_xlabel('$w$')
    ax2.bar(np.arange(1,5), N2, .9) 
    # ax1.grid(True)
    plt.sca(ax2)
    plt.xticks(np.arange(1, 5), ["u=1", "u=2", "u=3", "u=4"])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.set_title("(b) c=2", y = 1.0)
    ax2.set_ylim([0, N2[-1]])
    ax2.grid("on")

    # ax3.set_ylabel('# placements')
    # ax1.set_xlabel('$w$')
    ax3.bar(np.arange(1,5), N3, .9)
    # ax1.grid(True)
    plt.sca(ax3)
    plt.xticks(np.arange(1, 5), ["u=1", "u=2", "u=3", "u=4"])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax3.set_title("(c) c=3", y = 1.0)
    ax3.set_ylim([0, N3[-1]])
    ax3.grid("on")

    fig.tight_layout(pad=0.1)
    # plt.show()
    plt.savefig('./fig1.pdf', format='pdf', facecolor="white")

    N1 = [x / N1[-1] for x in N1]
    N2 = [x / N2[-1] for x in N2]
    N3 = [x / N3[-1] for x in N3]
    print(N1, N2, N3)
