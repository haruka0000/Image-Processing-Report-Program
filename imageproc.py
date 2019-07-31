import cv2
import numpy as np
import hough
import gaussian_newton

def main():
    img1 = cv2.imread("./input-2019.png")
    img2 = cv2.imread("./output-2019.png")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    theta, s = gaussian_newton.gaussianNewton(gray1, gray2, (0.4, 1.3), 0.005)
    print(theta, s)

    # imgproc(img1)
    angle = hough.get_degree(img1, img2)
    scale = hough.get_scale(img1, img2)
    # img1 = similarity(img1, angle, scale)
    print("hough angle =", np.rad2deg(angle), "[deg], hough scale =", scale)
    compare(img1, img2)
    # return
    kp1, kp2, matches = akaze(img1, img2)
    matches = sorted(matches, key=lambda x:x[0].distance)
    img1_pt = [list(map(int, kp1[m[0].queryIdx].pt)) for m in matches]
    img2_pt = [list(map(int, kp2[m[0].trainIdx].pt)) for m in matches]
    # img_surf = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:3], None, flags=2)
    img_akaze = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
    cv2.imwrite('akaze.png', img_akaze)
    res, af = imgproc(img1, img2, img1_pt, img2_pt)
    cv2.imwrite('estimate.png', res)
    compare(img2, res)
    ag, sc = calcAffineMatrix(af)
    reimg = similarity(img1, ag, sc)
    # compare(reimg, img2)


def similarity(img, angle, scale):
    height = img.shape[0]
    width = img.shape[1]
    center = (int(width/2), int(height/2))
    #getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D(center, angle , scale)
    #アフィン変換
    rotated_img = cv2.warpAffine(img, trans, (width,height))
    return rotated_img


def calcAffineMatrix(af):
    angle = np.arctan(af[0, 1] / af[0, 0])
    scale = af[0, 1] / np.sin(angle)
    print("angle =", np.rad2deg(angle), "[deg],", "scale =", scale)
    return angle, scale


def compare(img1, img2):
    img_hconcat = cv2.hconcat([img1, img2])
    cv2.imwrite("compare.png", img_hconcat)

    bgObg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgmask = bgObg.apply(img1)
    fgmask = bgObg.apply(img2)
    cv2.imwrite("diff.png", fgmask)


def imgproc(img1, img2, img1_pt, img2_pt):
    # src = np.float32(img1_pt[0:3])
    # dst = np.float32(img2_pt[0:3])
    src = []
    dst = []

    # 特徴点が同じになる点を排除
    for pt1, pt2 in zip(img1_pt, img2_pt):
        if pt1 not in src and pt2 not in dst:
            src.append(pt1)
            dst.append(pt2)
        if len(src) == 3:
            break
    src = np.float32(src)
    dst = np.float32(dst)
    
    af = cv2.getAffineTransform(src, dst)
    print(af)
    # M = cv2.getRotationMatrix2D((img2.shape[1]/2,img2.shape[0]/2),90,1)
    img = cv2.warpAffine(img1, af, (img2.shape[1], img2.shape[0]))
    return img, af
    pass


def surf(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    


    surf = cv2.xfeatures2d.SURF_create()
    kp1_surf, des1_surf = surf.detectAndCompute(gray1, None)
    kp2_surf, des2_surf = surf.detectAndCompute(gray2, None)

    bf_surf = cv2.BFMatcher()
    matches_surf = bf_surf.knnMatch(des1_surf, des2_surf, k=2)

    thr_surf = []
    for i, j in matches_surf:
        if i.distance < 0.4 * j.distance:
            thr_surf.append([i])

    img_surf = cv2.drawMatchesKnn(img1, kp1_surf, img2, kp2_surf, thr_surf, None, flags=2)
    # cv2.imwrite('pic2_out-match_surf.jpg', img_surf)
    return kp1_surf, kp2_surf, thr_surf

def akaze(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # ret,gray1 = cv2.threshold(gray1,254,255,cv2.THRESH_BINARY)
    # ret,gray2 = cv2.threshold(gray2,254,255,cv2.THRESH_BINARY)
    cv2.imwrite("test.png", gray1)

    # gray1 = cv2.Canny(gray1, 70, 90)
    # gray2 = cv2.Canny(gray2, 70, 90)
    # cv2.imwrite('edge.png', gray1)

    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(gray1, None)
    kp2, des2 = akaze.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches_surf = bf.knnMatch(des1, des2, k=2)

    thr = []
    for i, j in matches_surf:
        if i.distance < 0.5 * j.distance:
            thr.append([i])

    img_surf = cv2.drawMatchesKnn(img1, kp1, img2, kp2, thr, None, flags=2)
    # cv2.imwrite('pic2_out-match_surf.jpg', img_surf)
    return kp1, kp2, thr

if __name__ == "__main__":
    main()
    pass