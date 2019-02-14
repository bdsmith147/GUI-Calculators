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
from time import time
from scipy.optimize import curve_fit


def Sort(tab, delta_arr):
    '''Sorts the eigenenergies into continuous arrays; without this, they sometimes 
    get scrambled in the diagonalization process'''
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
    '''Returns an array representing the Raman hamiltonian for a given time step'''
    Ham = [[(k + 2)**2 - delta, Omega/2, 0], [Omega/2, k**2 - epsilon, Omega/2], [0, Omega/2, (k-2)**2 + delta]]
    return Ham

def Eigen(delta, Omega, epsilon, kArr):
    '''Diagonalizes the Raman hamiltonian for a given time step, returning 
    the eigenenergies'''
    values = [] 
    for k in kArr:
        vals = np.linalg.eigvals(Ham(k, delta, Omega, epsilon))
        values.append(vals)
    values = np.array(values)
    values = Sort(values, kArr)
    return np.array(values)
    
def parabola(x, A, x0, y0):
        return A**2 * (x - x0)**2 + y0
    

filename = 'AC_dispersion_calculations'
reset = 1 #Set to 1 if you want to force recalculate the data
try:
    #Tries to read the data from a file
    if not reset: 
        data = np.load(filename + '.npz')   
        kArr = data['kArr']
        values = data['vals']
        mins = data['mins']
        tArr = data['tArr']
        dArr = data['dArr']
        params = data['params']
    else:
        print('Going to redo the data...')
        raise Exception()
    
except:
    print('...calculating...')
    #Define quasimomentum array
    Nkpts = 100
    kMax = 4
    kArr = np.linspace(-kMax, kMax, Nkpts)
    
    #Define time array
    tMin, tMax = 0, 3
    Ntpts = 200 
    tArr = np.linspace(tMin, tMax, Ntpts)
    deltaMax = 5
    deltaArr = deltaMax*np.sin(2*np.pi*tArr)
    window = 20    
    
    #Define experimental parameters, and find eigenstates for each time step 
    Omeg = 10
    eps = 0
    values = np.empty((Ntpts, Nkpts, 3),dtype=float)
    mins = np.empty((Ntpts), dtype=int)
    params = np.empty((Ntpts, 3))
    for i, delt in enumerate(deltaArr):
        vals = Eigen(delt, Omeg, eps, kArr)
        minelem = np.argmin(vals[:,0])
        mins[i] = minelem
        values[i,:,:] = vals
        guess = [1, kArr[minelem], vals[minelem,0]]
        fitx, fity = kArr[minelem-window:minelem+window], vals[minelem-window:minelem+window,0]
        popt, pcov = curve_fit(parabola, fitx, fity, guess)
        params[i,:] = popt
        #print(popt)
    #Save data, so that you don't have to recalculate it each time.
    np.savez(filename, kArr=kArr, vals=values, mins=mins, tArr=tArr, dArr=deltaArr, params=popt)

finally:
    begtime = time()  
    #Initialize the figure
    fig1 = plt.figure(figsize=(6,9))
    fig1.subplots_adjust(top=1.2)
    
    #Set properties for the top subplot
    ax1 = fig1.add_subplot(2, 1, 1)
    dline, = ax1.plot(tArr[0], deltaArr[0], color='k', lw=2)
    ax1.set_title('AC Detuning \n($\Omega, \epsilon$) = (%i, %i) $E_L$'%(Omeg, eps))
    ax1.set_xlim(0, tMax)
    ax1.set_ylim(-deltaMax-1, deltaMax+1)
    ax1.set_xlabel('Time (au)')
    ax1.set_ylabel('Detuning ($\delta/E_L$)')
    
    #Set properties for the bottom subplot
    ax2 = fig1.add_subplot(2, 1, 2)
    l0, l1, l2, = ax2.plot(kArr, values[0,:,:], lw=2)
    s0 = ax2.scatter(kArr[mins[0]], values[0,mins[0],0], c='red', s=50)
    fit = ax2.plot(kArr, parabola(kArr, params*))
    ax2.set_xlabel('Quasimomentum ($q/k_L$)')
    ax2.set_ylabel('Energy ($E_L$)')
    
    plt.tight_layout()
    
    #Animate and save the function
    def update_lines(fnum, kArr, eigs, tArr, dArr, line0, line1, line2, s0, dline):
        '''This is the animator function and updates the plot at each frame'''
        line0.set_ydata(eigs[fnum,:,0])
        line1.set_ydata(eigs[fnum,:,1])
        line2.set_ydata(eigs[fnum,:,2])
        s0.set_offsets(np.hstack((kArr[mins[fnum], np.newaxis], 
                                  values[fnum,mins[fnum],0, np.newaxis]))) #This is a weird line, but it works.
        dline.set_data(tArr[:fnum], deltaArr[:fnum])
        return line0, line1, line2, s0,
    
    line_ani = animation.FuncAnimation(fig1, update_lines, Ntpts, 
                                       fargs=(kArr, values, tArr, deltaArr, l0, l1, l2, s0, dline), 
                                       interval=100, blit=True)
    line_ani.save('AC_dispersion_animation.mp4', writer='ffmpeg')
    print("Duration: ", time() - begtime)

