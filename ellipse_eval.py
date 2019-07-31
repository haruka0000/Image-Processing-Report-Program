import numpy as np
import matplotlib.pyplot as plt

def KCR(dat, param, sigma):
    f0 = 600
    u  = param
    M  = np.zeros((6, 6))
    L  = np.zeros((6, 6))

    for x, y in zip(dat[0], dat[1]):
        corr_xi = np.array([
            [x**2, x*y, 0, f0*x, 0, 0], 
            [x*y, (x**2)+(y**2), x*y, f0*y, f0*x, 0],
            [0, x*y, y**2, 0, f0*y, 0],
            [f0*x, f0*y, 0, f0**2, 0, 0],
            [0, f0*x, f0*y, 0, f0**2, 0],
            [0, 0, 0, 0, 0, 0]]
        )
        xi = np.array([x**2, 2*x*y, y**2, 2*f0*x, 2*f0*y, f0**2])
        M  = (xi.reshape(-1, 1) @ xi.reshape(1, -1)) / (u @ (corr_xi @ u)) + M
    
    la, v = np.linalg.eigh(M)
    la = sorted(la, reverse=True)

    D = sigma * np.sqrt(sum([(1 / lai) for lai in la[:5]]))

    return D


def BD(pred_params, true_param, n):
    u_true  = np.array(true_param)
    u       = [np.array(ui) for ui in pred_params]
    # print(u_true, u)

    I       = np.eye(len(true_param))
    uut     = np.dot(u_true.reshape(-1, 1), u_true.reshape(1, -1))
    D       = np.sqrt(sum([np.linalg.norm((I - uut) @ ui) for ui in u]) / n)    
    B       = np.linalg.norm(sum([((I - uut) @ ui) for ui in u]) / n)

    return B, D



def viewErrGraph(kcr, B, D, sigma):
    plt.plot(sigma, kcr)
    plt.plot(sigma, D[0])
    plt.plot(sigma, D[1])
    plt.show()