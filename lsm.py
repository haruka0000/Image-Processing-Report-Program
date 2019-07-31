import numpy as np
import math

def CircleFitting(x,y):
    """最小二乗法による円フィッティングをする関数
        input: x,y 円フィッティングする点群

        output  cxe 中心x座標
                cye 中心y座標
                re  半径

        参考
        一般式による最小二乗法（円の最小二乗法）　画像処理ソリューション
        http://imagingsolution.blog107.fc2.com/blog-entry-16.html
    """

    sumx  = sum(x)
    sumy  = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix,iy) in zip(x,y)])

    F = np.array([[sumx2,sumxy,sumx],
                  [sumxy,sumy2,sumy],
                  [sumx,sumy,len(x)]])

    G = np.array([[-sum([ix ** 3 + ix*iy **2 for (ix,iy) in zip(x,y)])],
                  [-sum([ix ** 2 *iy + iy **3 for (ix,iy) in zip(x,y)])],
                  [-sum([ix ** 2 + iy **2 for (ix,iy) in zip(x,y)])]])

    T=np.linalg.inv(F).dot(G)

    cxe=float(T[0]/-2)
    cye=float(T[1]/-2)
    re=math.sqrt(cxe**2+cye**2-T[2])
    #print (cxe,cye,re)
    return (cxe,cye,re)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    """Unit Test"""
    #推定する円パラメータ
    cx=4;   #中心x
    cy=10;  #中心y
    r=30;   #半径

    #円の点群の擬似情報
    plt.figure()
    x=range(-10,10);
    y=[]
    for xt in x:
        y.append(cy+math.sqrt(r**2-(xt-cx)**2))

    #円フィッティング
    (cxe,cye,re)=CircleFitting(x,y)

    #円描画
    theta=np.arange(0,2*math.pi,0.1)
    xe=[]
    ye=[]
    for itheta in theta:
        xe.append(re*math.cos(itheta)+cxe)
        ye.append(re*math.sin(itheta)+cye)
    xe.append(xe[0])
    ye.append(ye[0])

    plt.plot(x,y,"ob",label="raw data")
    plt.plot(xe,ye,"-r",label="estimated")
    plt.plot(cx,cy,"xb",label="center")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()
