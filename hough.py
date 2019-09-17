import cv2
import numpy as np
import math
import sys
import copy
# from scipy import ndimage

# 画像の傾き検出
# @return 水平からの傾き角度
def get_degree(img_1, img_2):
    # img_1 = cv2.imread(filename_01)
    deg_1 = calc_degree(img_1, "01")
    
    # img_2 = cv2.imread(filename_02)
    deg_2 = calc_degree(img_2, "02")

    deg_dif = deg_1 - deg_2
    # print(deg_dif)

    # horizontal_img_1 = ndimage.rotate(img_1, deg_1)
    # cv2.imwrite('horizontal_' + filename_01 + '.png', horizontal_img_1)
    # horizontal_img_2 = ndimage.rotate(img_2, deg_2)
    # cv2.imwrite('horizontal_' + filename_02 + '.png', horizontal_img_2)
    
    # オリジナル（線が描画されたいない）
    # img_1 = cv2.imread(filename_01)
    # rotate_img = rotate(img_1, deg_dif, 1.0)
    # cv2.imwrite('hough_rotate.png',rotate_img)

    return deg_dif


def calc_degree(org_img, filename): 
    img = copy.deepcopy(org_img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 150, apertureSize = 3)
    
    minLineLength = 200
    maxLineGap = 30

    # ハフ変換で得られた直線
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    
    sum_deg = 0;
    count = 0;

    for line in lines:
        for x1,y1,x2,y2 in line:
            # 線分の座標から直線を作成し，角度を算出
            deg = math.degrees(math.atan2((y2-y1), (x2-x1)))
            
            # 複数の線分から得られた角度の平均用
            sum_deg += deg;
            count += 1

            # 利用したラインを描画
            cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)

    # 角度の平均
    ave_deg = sum_deg / count 
    cv2.imwrite('houghlines_'+filename+'.jpg',img)

    return ave_deg


def get_scale(img1, img2):
    # img1 = cv2.imread(file_01)
    # img2 = cv2.imread(file_02)
    
    param_01 = get_r(img1, "01", 10, 300)
    param_02 = get_r(img2, "02", 10, 90)
    
    limit = len(param_01)
    if len(param_01) > len(param_02):
        limit = len(param_02)

    scale_list = []

    for p1, p2 in zip(param_01[:limit], param_02[:limit]):
        scale = p2[2] / p1[2]
        scale_list.append(scale)
    
    scale_ave = sum(scale_list) / len(scale_list)
    return scale_ave

def get_r(img_org, name, thr01, thr02):
    img = copy.deepcopy(img_org)
    img = cv2.medianBlur(img,5)
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (9,9))
    # edges = cv2.Canny(img, 50, 110, apertureSize = 3)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,10, param1=thr01, param2=thr02, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    
    params = []
    for i in circles[0,:]:
        # draw the outer circle
        # cv2.circle(img_org,(i[0],i[1]), i[2], (0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]), i[2], (0,255,0),3)
        params.append(i)
    cv2.imwrite('houghcircles_' + name + '.png', img)

    return params

if __name__=="__main__":
    filename_01 = sys.argv[1]
    filename_02 = sys.argv[2]
    get_degree(filename_01, filename_02)
    
    