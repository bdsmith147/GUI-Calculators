# -*- coding: utf-8 -*-
"""
Created on Fri Feb 08 14:44:41 2019

@author: Lindsay LeBlanc
"""

import numpy as np
from matplotlib import pyplot as plt
Npts = 100
kMax = 4
kArr = np.linspace(-kMax, kMax, Npts)

def Sort(tab, delta_arr):
    length = tab.shape[0]
    width = tab.shape[1]
    Del = delta_arr[1] - delta_arr[0]
    for col in range(width):
        for row in range(2, length):
#            dist = np.abs(tab[row,:] - tab[row-1, col])
            slope = (tab[row-1, col] - tab[row-2, col])/Del
            guess = slope*Del + tab[row-1, col]
            dist = np.abs(tab[row,:] - guess)
            elem = np.argmin(dist)
            if elem != col:
                tab[row, col], tab[row, elem] = tab[row, elem], tab[row, col]
    return tab

def Ham(k, delta, Omega, epsilon):
    Ham = [[(k + 2)**2 - delta, Omega/2, 0], [Omega/2, k**2 - epsilon, Omega/2], [0, Omega/2, (k-2)**2 + delta]]
    return Ham

def Eigen(delta, Omega, epsilon):
    values = [] 
    for k in kArr:
        vals = np.linalg.eigvals(Ham(k, delta, Omega, epsilon))
        values.append(vals)
    values = np.array(values)
    values = Sort(values, kArr)
    return np.array(values)
    

#try:
#     pass   
#    
#except:
Omeg = 10
delt = -2
eps = 0
values = Eigen(delt, Omeg, eps)
minElem = np.argmin(values[:,0])
plt.figure()
plt.plot(kArr, values, lw=2)
plt.scatter(kArr[minElem], values[minElem,0], c='red', s=100)
plt.show()