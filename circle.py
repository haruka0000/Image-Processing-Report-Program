import cv2
import numpy as np

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
    img = cv2.medianBlur(img_org,5)
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (9,9))
    # edges = cv2.Canny(img, 50, 110, apertureSize = 3)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,10, param1=thr01, param2=thr02, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    
    params = []
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img_org,(i[0],i[1]), i[2], (0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]), 2, (0,0,255),3)
        params.append(i)
    cv2.imwrite('detected_' + name + '.png', img_org)

    return params

if __name__ == '__main__':
    print(get_scale('input-2019.png', 'output-2019.png'))
    