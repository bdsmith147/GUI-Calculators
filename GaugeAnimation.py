# -*- coding: utf-8 -*-
"""
Created on Fri Feb 08 14:44:41 2019

@author: Lindsay LeBlanc
"""
#Goals Saturday 20190209:
#   1. Get the code to output and save as np arrays multiple dispersions based on an
#   AC detuning
#   2. Load the np arrays and animate the graph; show the AC detuning at
#   the same time as another subplot
#   3. Run this code on a remote machine, i.e. Joseph's. Learn how to do that well

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


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

def Eigen(delta, Omega, epsilon, kArr):
    values = [] 
    for k in kArr:
        vals = np.linalg.eigvals(Ham(k, delta, Omega, epsilon))
        values.append(vals)
    values = np.array(values)
    values = Sort(values, kArr)
    return np.array(values)
    

filename = 'Test'
reset = 0
try:
    #See if data has already been calculated, or if user desires to do it again
    if not reset: 
        data = np.load(filename + '.npz')   
        kArr = data['kArr']
        values = data['vals']
        mins = data['mins']
        tArr = data['tArr']
        dArr = data['dArr']
    else:
        raise Exception('Going to redo the data...')
    
except:
    print('...recalculating...')
    Nkpts = 100
    kMax = 4
    kArr = np.linspace(-kMax, kMax, Nkpts)
    
    
    tMin, tMax = 0, 3
    Ntpts = 100
    tArr = np.linspace(tMin, tMax, Ntpts)
    deltaMax = 5
    deltaArr = deltaMax*np.sin(2*np.pi*tArr)
    
    plt.figure()
    plt.plot(tArr, deltaArr, color='C0')
    plt.show()
    
    Omeg = 10
    eps = 0
    values = np.empty((Ntpts, Nkpts, 3),dtype=float)
    mins = np.empty((Ntpts), dtype=int)
    for i, delt in enumerate(deltaArr):
        vals = Eigen(delt, Omeg, eps, kArr)
        mins[i] = np.argmin(vals[:,0])
        values[i,:,:] = vals
        
    np.savez(filename, kArr=kArr, vals=values, mins=mins, tArr=tArr, dArr=deltaArr)

finally:
#    plt.figure()
#    for vals in values:
#        plt.plot(kArr, vals)
#    plt.show()
    
    def update_lines(fnum, kArr, eigs, line0, line1, line2, s0):
        line0.set_ydata(eigs[fnum,:,0])
        line1.set_ydata(eigs[fnum,:,1])
        line2.set_ydata(eigs[fnum,:,2])
        s0.set_offsets(np.hstack((kArr[mins[fnum], np.newaxis], values[fnum,mins[fnum],0, np.newaxis])))
        return line0, line1, line2, s0,
        

        
    fig1 = plt.figure()
    init = np.full(Nkpts, np.nan)
    l0, l1, l2, = plt.plot(kArr, values[0,:,:])
    s0 = plt.scatter(kArr[mins[0]], values[0,mins[0],0], c='red')
    plt.xlabel('Quasimomentum ($q/k_L$)')
    plt.ylabel('Energy ($E_L$)')
    
#    def init():
#        l0.set_ydata([np.nan]*len(kArr))
#        l1.set_ydata([np.nan]*len(kArr))
#        l2.set_ydata([np.nan]*len(kArr))
#        return l0, l1, l2,
    
    line_ani = animation.FuncAnimation(fig1, update_lines, Ntpts, fargs=(kArr, values, l0, l1, l2, s0), interval=100, blit=True)
    line_ani.save('lines.mp4')
    
    

