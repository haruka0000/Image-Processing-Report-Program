import numpy as np
import cv2
import imageproc


"""
J = 1/2 \sum(I'(x', y') - I(x, y))^2
"""
def gaussianNewton(img1, img2, init, ep):
    theta = init[0]
    s = init[1]
    while True:
        img1_ = imageproc.similarity(img1, theta, s)
        img2_ = imageproc.similarity(img2, -theta, 1/s)
        # img_diff = np.array(img2_) - np.array(img1)
        bgObg = cv2.bgsegm.createBackgroundSubtractorMOG()
        img_diff = bgObg.apply(img1_)
        img_diff = bgObg.apply(img2)
        cv2.imshow("img2_trans", img1_)
        cv2.imshow("img1", img1)    
        # cv2.imshow("x_diff", x_diff)
        # cv2.imshow("y_diff", y_diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        J = np.power(img_diff, 2).sum() / 2
        print("theta =", np.rad2deg(theta), "[deg], scale =", s, "objective func =", J)
        if (J < ep):
            break
        x_diff, y_diff = differential(img2_)
        # img_change = imageproc.similarity(img1, theta, s)
        # bgObg = cv2.bgsegm.createBackgroundSubtractorMOG()
        # img_diff = bgObg.apply(img_change)
        # img_diff = bgObg.apply(img2)
        # cv2.imshow("diff", img_diff)
        # cv2.imshow("diff_2", img2 - img1)    
        # cv2.imshow("x_diff", x_diff)
        # cv2.imshow("y_diff", y_diff)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img_idx = [np.array(list(range(0, img1.shape[0])), dtype=int), np.array(list(range(0, img1.shape[1])), dtype=int)]

        # dx_dtheta = img_idx[1] * s * (-np.sin(theta)) + img_idx[0] * s * (-np.cos(theta))
        # dy_dtheta = img_idx[1] * s * (np.cos(theta)) + img_idx[0] * s * (-np.sin(theta))
        # dx_ds = img_idx[1] * np.cos(theta) + img_idx[0] *  (-np.sin(theta))
        # dy_ds = img_idx[1] * np.sin(theta) + img_idx[0] * np.cos(theta)
        # dx_dtheta_ds = img_idx[1] * (-np.sin(theta)) + img_idx[0] * (-np.cos(theta))
        # dy_dtheta_ds = img_idx[1] * (np.cos(theta)) + img_idx[0] * (-np.sin(theta))

        dx_dtheta_rotate = np.array([[-s * np.sin(theta), -s * np.cos(theta)], [s * np.cos(theta), -s * np.sin(theta)]])
        dx_ds_rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        dx_dtheta_ds_rotate = np.array([[-np.sin(theta), -np.cos(theta)], [np.cos(theta), -np.sin(theta)]])
        
        # dx_dtheta_rotate = np.linalg.inv(dx_dtheta_rotate)
        # dx_ds_rotate = np.linalg.inv(dx_ds_rotate)
        # dx_dtheta_ds_rotate = np.linalg.inv(dx_dtheta_ds_rotate)

        # dx_dtheta = np.zeros((2, img1.shape[0], img1.shape[1]))
        # dx_ds = np.zeros((2, img1.shape[0], img1.shape[1]))
        # dx_dtheta_ds = np.zeros((2, img1.shape[0], img1.shape[1]))
        # img_tmp1 = img2
        # img_tmp2 = img2
        # img_tmp3 = img2
        
        dx_dtheta = np.array([np.dot(dx_dtheta_rotate, np.array([i, j])) for i in range(img1.shape[1]) for j in range(img1.shape[1])]).reshape(2, 950, 950)
        dx_ds = np.array([np.dot(dx_ds_rotate, np.array([i, j])) for i in range(img1.shape[1]) for j in range(img1.shape[1])]).reshape(2, 950, 950)
        dx_dtheta_ds = np.array([np.dot(dx_dtheta_ds_rotate, np.array([i, j])) for i in range(img1.shape[1]) for j in range(img1.shape[1])]).reshape(2, 950, 950)
        # for i in range(img1.shape[1]):
        #     for j in range(img1.shape[0]):
        #         dx_dtheta[:, i, j] = np.dot(dx_dtheta_rotate, np.array([i, j]))
        #         dx_ds[:, i, j] = np.dot(dx_ds_rotate, np.array([i, j]))
        #         dx_dtheta_ds[:, i, j] = np.dot(dx_dtheta_ds_rotate, np.array([i, j]))
                # if np.all(dx_dtheta < img1.shape[0]) and np.all(dx_ds < img1.shape[0]) and np.all(dx_dtheta_ds < img1.shape[0]):
                #     img_tmp1[i, j] = img1[int(dx_dtheta[1, i, j]), int(dx_dtheta[0, i, j])]
                #     img_tmp2[i, j] = img1[int(dx_ds[1, i, j]), int(dx_ds[0, i, j])]
                #     img_tmp3[i, j] = img1[int(dx_dtheta_ds[1, i, j]), int(dx_dtheta_ds[0, i, j])]

        # J_theta, J_theta-theta, J_s, J_s-s, J_theta-s
        # img_diff = cv2.add(img2, -img1 - 1)

        # cv2.imwrite("diff.png", fgmask)
        # cv2.imshow("diff", img_diff)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img_diff[100, 500])
        # print(-img1 - 1)
        # print(img2[100, 500])
        # img_diff = img2 - img1
        theta_diff = ((img_diff) * (x_diff * dx_dtheta[0] + y_diff * dx_dtheta[1])).sum()
        theta_diff2 = np.power(x_diff * dx_dtheta[0] + y_diff * dx_dtheta[1], 2).sum()
        s_diff = ((img_diff) * (x_diff * dx_ds[0] + y_diff * dx_ds[1])).sum()
        s_diff2 = np.power(x_diff * dx_ds[0] + y_diff * dx_ds[1], 2).sum()
        theta_s_diff = ((x_diff * dx_ds[0] + y_diff * dx_ds[1]) * (x_diff * dx_dtheta[0] + y_diff * dx_dtheta[1]) + img_diff * (x_diff * dx_dtheta_ds[0] + y_diff * dx_dtheta_ds[1])).sum()
        
        # theta_diff = ((img2 - img1) * (x_diff * dx_dtheta[0] + y_diff * dx_dtheta[1])).sum()
        # theta_diff2 = np.power(x_diff * dx_dtheta[0] + y_diff * dx_dtheta[1], 2).sum()
        # s_diff = ((img2 - img1) * (x_diff * dx_ds[0] + y_diff * dx_ds[1])).sum()
        # s_diff2 = np.power(x_diff * dx_dtheta[0] + y_diff * dx_dtheta[1], 2).sum()
        # theta_s_diff = ((img2 - img1) * (x_diff * dx_ds[0] + y_diff * dx_ds[1]) * (x_diff * dx_dtheta[0] + y_diff * dx_dtheta[1]) + (img2 - img1) * (x_diff * dx_dtheta_ds[0] + y_diff * dx_dtheta_ds[1])).sum()
        
        Jacobi = np.array([[theta_diff2, theta_s_diff], [theta_s_diff, s_diff2]])
        Jacobi_inv = np.linalg.inv(Jacobi)
        # print(J_inv)
        del_theta = -Jacobi_inv[0, 0] * theta_diff - Jacobi_inv[0, 1] * s_diff
        del_s = -Jacobi_inv[1, 0] * theta_diff - Jacobi_inv[1, 1] * s_diff

        
        theta = theta + del_theta
        s = s + del_s
        # img1 = imageproc.similarity(img1, del_theta, del_s)
        # img_diff = bgObg.apply(img_change)
        # img_diff = bgObg.apply(img2)
        # objective fuction
        # img_diff = img2 - img1

    return theta, s
    

def differential(img):
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    return sobelx, sobely




