import numpy as np
import matplotlib.pyplot as plt
import calc_ellipse
import math
from ellipse_eval import *
from plot import *

# least squares method
def LSM(dat):
    f0 = 600
    x = dat[0]
    y = dat[1]
    _2x   = [2 * f0 * ix for ix in x]
    _2y   = [2 * f0 * iy for iy in y]
    _x2   = [ix ** 2 for ix in x]
    _y2   = [iy ** 2 for iy in y]
    _2xy   = [2 * ix * iy for (ix,iy) in zip(x,y)]

    M = []
    for i in range(len(x)):
        tmp_xi = np.array([_x2[i], _2xy[i], _y2[i], _2x[i], _2y[i], f0**2])
        M.append(np.dot(tmp_xi.reshape(-1, 1), tmp_xi.reshape(1, -1)))
    
    sum_M = M[0]
    for i in range(1, len(M)):
        sum_M = sum_M + M[i]
    
    eigen_value, eigen_vector = np.linalg.eig(sum_M)
    pred_param = eigen_vector[:, eigen_value.shape[0]-1]

    return pred_param


# maximum likelihood estimation
def MLE(dat, param, ep):
    u = param
    f0 = 600
    k = 0
    c = 0
    for k in range(100):
        M = np.zeros((6, 6))
        L = np.zeros((6, 6))
        N = np.zeros((6, 6))
        for x, y in zip(dat[0], dat[1]):
            corr_xi = np.array([
                [x**2, x*y      , 0,    f0*x,  0,     0], 
                [x*y,  x**2+y**2, x*y,  f0*y,  f0*x,  0],
                [0,    x*y,       y**2, 0,     f0*y,  0],
                [f0*x, f0*y,      0,    f0**2, 0,     0],
                [0,    f0*x,      f0*y, 0,     f0**2, 0],
                [0,    0,         0,    0,     0,     0]]
            )
            xi = np.array([x**2, 2*x*y, y**2, 2*f0*x, 2*f0*y, f0**2])
            xixi = xi.reshape(-1, 1) @ xi.reshape(1, -1)
            ucu = u  @ (corr_xi @ u)
            M = xixi / ucu + M
            L = (u @ xi)**2 * corr_xi / ucu**2 + L
            N = corr_xi / ucu + N

        # if ((M - L) == (M - L).T).all() == False:
        #     print("err!")
        #     raise Exception("error! A is not symmetric matrix!")

        la, v = np.linalg.eigh(M - L)
        min_v = np.array(v[:,0])
        
        if np.linalg.norm(min_v - u) < ep:
            u = min_v
            # print("la =", la, "\nv =", v)
            break
        u = min_v

    return u


def setErrND(dat, sigma):
    x = dat[0]
    y = dat[1]
    x = np.random.normal(scale=sigma, size=x.shape) + x
    y = np.random.normal(scale=sigma, size=y.shape) + y
    return x, y


def viewEllipse(dat, true_param, est_param):
    x = dat[0]
    y = dat[1]
    tvalid, taxis, tcenterEst, tRest = calc_ellipse.getEllipseProperty(true_param[0], true_param[1], true_param[2], true_param[3], true_param[4], true_param[5])
    tdataEst = calc_ellipse.generateVecFromEllipse(taxis, tcenterEst, tRest)
    evalid, eaxis, ecenterEst, eRest = calc_ellipse.getEllipseProperty(est_param[0], est_param[1], est_param[2], est_param[3], est_param[4], est_param[5])
    edataEst = calc_ellipse.generateVecFromEllipse(eaxis, ecenterEst, eRest) 
    # plt.plot(x, y)
    # fig, ax = plt.subplots(ncols = 1, figsize=(10, 10))
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    plt.plot(tdataEst[:, 0], tdataEst[:, 1])
    plt.plot(edataEst[:, 0], edataEst[:, 1])
    plt.show()


def exec(point_x, point_y, t_param):
    sigma = [0.01*i for i in range(100)]
    kcr = [KCR([point_x, point_y], t_param, sigmai) for sigmai in sigma]
    lsm_B_list = []
    lsm_D_list = []
    mle_B_list = []
    mle_D_list = []
    for si in sigma:
        # print(len(x))
        #円フィッティング
        
        n = 500
        params_lsm = []
        params_mle = []

        for i in range(n):
            x, y = setErrND([point_x, point_y], si)
            p = np.random.normal(size=param.shape)
            p = p / np.linalg.norm(p)
            params_lsm.append(LSM([x, y]))
            params_mle.append(MLE([x, y], p, ep=1.0e-6))
        # print(params_lsm)
        lsm_B, lsm_D = BE(params_lsm, t_param, n)
        mle_B, mle_D = BE(params_mle, t_param, n)
        lsm_B_list.append(lsm_B)
        lsm_D_list.append(lsm_D)
        mle_B_list.append(mle_B)
        mle_D_list.append(mle_D)

    # viewErrGraph(kcr, [lsm_B_list, mle_B_list], [lsm_D_list, mle_D_list], sigma)
    plot(sigma, lsm_D_list, mle_D_list, kcr, lsm_B_list, mle_B_list)


        # print("真値\t\t", true_param)
        # print("最小固有ベクトル",  eigen_vec)



if __name__ == "__main__":
    point_x, point_y = np.loadtxt("./points.dat", unpack=True)
    param = np.loadtxt("./true_param.dat", unpack=True)
    exec(point_x, point_y, param)
    # sigma = 1
    # x, y = setErrND([point_x, point_y], sigma)
    # p = np.random.normal(size=param.shape)
    # # p = param
    # p = p / np.linalg.norm(p)
    # print("init =", p)
    # u_lsm = LSM([x, y])
    # u_mle = MLE([point_x, point_y], p, 1.0e-6)
    # print("\nesm_param =", u_lsm, "\ntrue_param=", param)
    # print("itr =", itr+1, "\nesm_param =", u_mle, "\ntrue_param=", param)
    # viewEllipse([x, y], param, u_mle)
    # sigma = [0.1*i for i in range(10)]
    # err = [KCR([point_x, point_y], param, sigmai) for sigmai in sigma]
    # viewErrGraph(err, sigma)
    pass