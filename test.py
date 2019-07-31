# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy as sp

def createMMat(data, weight):
  
  dataMod = np.matrix(np.zeros((data.shape[0], 6)))
  for i in range(data.shape[0]):
    dataMod[i, 0] = data[i, 0]**2
    dataMod[i, 1] = data[i, 0] * data[i, 1]
    dataMod[i, 2] = data[i, 1]**2
    dataMod[i, 3] = data[i, 0] * data[i, 2]
    dataMod[i, 4] = data[i, 1] * data[i, 2]
    dataMod[i, 5] = data[i, 2]**2
  
  M = np.zeros((6, 6))
  M = np.matrix(M)
  for i in range(dataMod.shape[0]):
    dM = dataMod[i, :].T * dataMod[i, :]
    M = M + weight[i, 0] * dM

  return M / dataMod.shape[0]

def createLMat(data, weight, theta, covs):
  
  dataMod = np.matrix(np.zeros((data.shape[0], 6)))
  for i in range(data.shape[0]):
    dataMod[i, 0] = data[i, 0]**2
    dataMod[i, 1] = data[i, 0] * data[i, 1]
    dataMod[i, 2] = data[i, 1]**2
    dataMod[i, 3] = data[i, 0] * data[i, 2]
    dataMod[i, 4] = data[i, 1] * data[i, 2]
    dataMod[i, 5] = data[i, 2]**2
  
  L = np.matrix(np.zeros((6, 6)))
  for i in range(dataMod.shape[0]):
    coeff = weight[i, 0]**2 * (dataMod[i, :] * theta)**2
    L = L + coeff[0, 0] * covs[i]
  
  return L / dataMod.shape[0]

def fittingEllipseWithLS(data, normTerm):
  
  weight = np.matrix(np.ones(data.shape[0])).T
  dataMod = np.concatenate((data / normTerm, np.ones((data.shape[0], 1))), axis=1)
  M = createMMat(dataMod, weight)  
  lamdas, v = np.linalg.eigh(M)
  #print(lamdas)
  #print(v)

  theta = v[:, np.argmin(np.absolute(lamdas))]
  #print(theta)
  
#   lamdas, v = np.linalg.eigh(M)
  #print(lamdas)
  #print(v)

#   theta = np.matrix(v[:, np.argmin(np.absolute(lamdas))]).T
  #print(theta)
  theta[3, 0] = normTerm * theta[3, 0]
  theta[4, 0] = normTerm * theta[4, 0]
  theta[5, 0] = normTerm**2 * theta[5, 0]
  return theta

  
def fittingEllipseWithTaubin(data, normTerm):

  # Add normalized term.
  dataMod = np.concatenate((data / normTerm, np.ones((data.shape[0], 1))), axis=1)
    
  # Param Vector
  theta = np.matrix(np.zeros(6)).T
  
  # Covars
  covavg = np.matrix(np.zeros((6, 6)))
  for i in range(dataMod.shape[0]):
    data = dataMod[i, :]
    covavg = covavg + createCovMat(data)
  covavg = covavg / dataMod.shape[0]

  weight = np.ones(dataMod.shape[0])
  weight = np.matrix(weight).T
  M = createMMat(dataMod, weight);
  lamdas, v = np.linalg.eig(M)
 
  theta = np.matrix(v)[:, np.argmax(np.absolute(lamdas))]
  
  theta[3, 0] = normTerm * theta[3, 0]
  theta[4, 0] = normTerm * theta[4, 0]
  theta[5, 0] = normTerm**2 * theta[5, 0]

  return theta  
  
           
def createCovMat(data):
  
  x = data[0, 0]
  y = data[0, 1]
  f0 = data[0, 2]
  xx = x**2
  yy = y**2
  xy = x*y
  f0x = f0*x
  f0y = f0*y
  f0f0 = f0**2
  
  cov = np.matrix([[xx,  xy,     0,   f0x,     0,    0], \
                   [xy,  xx+yy, xy,   f0y,   f0x,    0], \
                   [0,   xy,    yy,     0,   f0y,    0], \
                   [f0x, f0y,   0,    f0f0,    0,    0], \
                   [0,   f0x,   f0y,  0,     f0f0,   0], \
                   [0,   0,     0,    0,     0,      0]])
  
  return cov
           
def fittingEllipseWithItrReweight(data, normTerm):
  
  # Add normalized term.
  dataMod = np.concatenate((data / normTerm, np.ones((data.shape[0], 1))), axis=1)
  
  # Param Vector
  theta = np.matrix(np.zeros(6)).T
  thetaOrg = theta
  
  # Weight matrix.
  weight = np.ones(dataMod.shape[0])
  weight = np.matrix(weight).T

  # Covars
  covs = []
  for i in range(dataMod.shape[0]):
    data = dataMod[i, :]
    covs.append(createCovMat(data))

  loop = 0
  while True:
    
    # M Matrix
    M = createMMat(dataMod, weight)
    lamdas, v = np.linalg.eigh(M)
    thetaOrg = theta
    theta = v[:, np.argmin(np.absolute(lamdas))]
    
    term = np.linalg.norm(theta - thetaOrg)  
    if term < 0.0001 or loop > 20:
      break

    for i in range(dataMod.shape[0]):
      alp = theta.T * covs[i] * theta
      weight[i, 0] = 1 / (alp)
    
    loop = loop + 1
  
  theta[3, 0] = normTerm * theta[3, 0]
  theta[4, 0] = normTerm * theta[4, 0]
  theta[5, 0] = normTerm**2 * theta[5, 0]    
  return theta          

          
  
def fittingEllipseWithFNS(data, normTerm):
  
    # Add normalized term.
  dataMod = np.concatenate((data / normTerm, np.ones((data.shape[0], 1))), axis=1)
  
  # Param Vector
  theta = np.matrix(np.zeros(6)).T
  thetaNew = theta
  thetaOrg = theta
  
  # Weight matrix.
  weight = np.ones(dataMod.shape[0])
  weight = np.matrix(weight).T

  # Covars
  covs = []
  for i in range(dataMod.shape[0]):
    data = dataMod[i, :]
    covs.append(createCovMat(data))

  loop = 0
  while True:
    
    # M Matrix
    M = createMMat(dataMod, weight)
    L = createLMat(dataMod, weight, theta, covs)
    lamdas, v = np.linalg.eigh(M - L)
    thetaOrg = theta
    thetaNew = v[:, np.argmin(np.absolute(lamdas))]
    #theta = (thetaNew + thetaOrg) / 2
    theta = thetaNew
    #theta = theta / np.linalg.norm(theta)
    
    term = np.linalg.norm(theta - thetaOrg)  
    if term < 0.0001 or loop > 30:
      break

    for i in range(dataMod.shape[0]):
      alp = theta.T * covs[i] * theta
      weight[i, 0] = 1 / (alp)
    
    loop = loop + 1

  theta[3, 0] = normTerm * theta[3, 0]
  theta[4, 0] = normTerm * theta[4, 0]
  theta[5, 0] = normTerm**2 * theta[5, 0]  
  return theta
  
def plotData(dictDataPlot, dictDataScatter):
  
  fig, ax = plt.subplots(ncols = 1, figsize=(10, 10))
  plt.xlim(0, 10)
  plt.ylim(0, 10)
  
  for key in dictDataPlot:
    ax.plot(dictDataPlot[key][:, 0], dictDataPlot[key][:, 1], linewidth=1, label=key)
  
  for key in dictDataScatter:    
    ax.scatter(dictDataScatter[key][:, 0], dictDataScatter[key][:, 1], s = 2)
  
  ax.legend()

    
def addGaussError(data, avg, stdv, absmax):
  noise = np.random.normal(avg, stdv, data.shape)
  noise = np.clip(noise, -(absmax + avg), absmax + avg) 
  dataWError = data + noise
  return dataWError
 
if __name__ == "__main__":
  
  # Importing Library for Testing.
  import sys
  sys.path.append('../')
  from calc_ellipse import generateVecFromEllipse
  from calc_ellipse import getEllipseProperty
  
  print("Ellipse Fitting Sample")
  
  # Define ellipse property.
  axisX = 3
  axisY = 3
  centerX = 5
  centerY = 5
  rad = math.radians(45)
  
  # Prepare argumetns.
  axisOrg = np.matrix([[axisX], [axisY]])
  centOrg = np.matrix([[centerX], [centerY]])
  Rorg = np.matrix([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
  
  # Generating Data
  dataOrg = generateVecFromEllipse(axisOrg, centOrg, Rorg)
  dataLimited = generateVecFromEllipse(axisOrg, centOrg, Rorg, [-2*math.pi/6 * 1, 2*math.pi/6], 200)

  # Add Noise
  dataNoised = addGaussError(dataLimited, 0, 0.03, 10.1)
  
  # Least Square Fitting
  pLs = fittingEllipseWithLS(dataNoised, max(centerX + axisX, centerY + axisY))
  print(pLs)
  
  # Generate Estimated Ellipse Property
  validLs, axisEstLs, centEstLs, TEstLs = \
  getEllipseProperty(pLs[0, 0], pLs[1, 0], pLs[2, 0], pLs[3, 0], pLs[4, 0], pLs[5, 0]) 
  
  # Iterative Reweight
  pItr = fittingEllipseWithItrReweight(dataNoised, max(centerX + axisX, centerY + axisY))
  #print(pItr)
  
  # Generate Estimated Ellipse Property
  validItr, axisEstItr, centEstItr, TEstItr = \
  getEllipseProperty(pItr[0, 0], pItr[1, 0], pItr[2, 0], pItr[3, 0], pItr[4, 0], pItr[5, 0]) 
    
  # Iterative Reweight
  pFns = fittingEllipseWithFNS(dataNoised, max(centerX + axisX, centerY + axisY))
  #print(pFns)
  
  # Generate Estimated Ellipse Property
  validFns, axisEstFns, centEstFns, TEstFns = \
  getEllipseProperty(pFns[0, 0], pFns[1, 0], pFns[2, 0], pFns[3, 0], pFns[4, 0], pFns[5, 0]) 
  
  # Taubin Method
  pTb = fittingEllipseWithTaubin(dataNoised, max(centerX + axisX, centerY + axisY))
  #pTb = fittingEllipseWithTaubin(dataNoised, 1)
  print(pTb)
  
  # Generate Estimated Ellipse Property
  validTb, axisEstTb, centEstTb, TEstTb = \
  getEllipseProperty(pTb[0, 0], pTb[1, 0], pTb[2, 0], pTb[3, 0], pTb[4, 0], pTb[5, 0])
  
  # Estimating Data
  dataEstLs = generateVecFromEllipse(axisEstLs, centEstLs, TEstLs)
  dataEstItr = generateVecFromEllipse(axisEstItr, centEstItr, TEstItr)
  dataEstFns = generateVecFromEllipse(axisEstFns, centEstFns, TEstFns)
  dataEstTb = generateVecFromEllipse(axisEstTb, centEstTb, TEstTb)
  
  dictData = {'Original' : dataOrg,\
              'Least SQ' : dataEstLs,\
              'Itr Weight' : dataEstItr,\
              'FNS' : dataEstFns, \
              'Taubin' : dataEstTb}
              
  dictDataScatter = {'Noised Sample' : dataNoised}
  
  plotData(dictData, dictDataScatter)
    