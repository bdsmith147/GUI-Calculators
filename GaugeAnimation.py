# -*- coding: utf-8 -*-
"""
Created on Fri Feb 08 14:44:41 2019

@author: Lindsay LeBlanc
"""
#Goals:
#   1. Plot the kmin, m* and Emin as a function of time
#   2. Run this code on a remote machine, i.e. Joseph's. Learn how to do that well. Retrieve the results.

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from time import time
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec


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
        return A * (x - x0)**2 + y0
    

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
    Nkpts = 200
    kMax = 4
    kArr = np.linspace(-kMax, kMax, Nkpts)
    
    #Define time array
    tMin, tMax = 0, 3
    Ntpts = 200 
    tArr = np.linspace(tMin, tMax, Ntpts)
    deltaMax = 1
    deltaArr = deltaMax*np.sin(2*np.pi*tArr)
    window = 50    
    
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
    np.savez(filename, kArr=kArr, vals=values, mins=mins, tArr=tArr, dArr=deltaArr, params=params)

finally:
    begtime = time()  
    #Initialize the figure
    fig1 = plt.figure(figsize=(9,9))
#    fig1.subplots_adjust(top=1.2)
    gs = gridspec.GridSpec(1, 2, figure=fig1)
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0])
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    
    ax1 = plt.Subplot(fig1, gs0[0])
    ax2 = plt.Subplot(fig1, gs0[1:])
    
    ax3 = plt.Subplot(fig1, gs1[0])
    ax4 = plt.Subplot(fig1, gs1[1])
    ax5 = plt.Subplot(fig1, gs1[2])
    fig1.add_subplot(ax1)
    fig1.add_subplot(ax2)
    fig1.add_subplot(ax3)
    fig1.add_subplot(ax4)
    fig1.add_subplot(ax5)
    
    #Set properties for the top subplot
    #ax1 = fig1.add_subplot(2, 1, 1)
    dline, = ax1.plot(tArr[0], deltaArr[0], color='k', lw=2)
    ax1.set_title('AC Detuning \n($\Omega, \epsilon$) = (%i, %i) $E_L$'%(Omeg, eps))
    ax1.set_xlim(0, tMax)
    ax1.set_ylim(-deltaMax-1, deltaMax+1)
    ax1.set_xlabel('Time (au)')
    ax1.set_ylabel('Detuning ($\delta/E_L$)')
    
    #Set properties for the bottom subplot
    #ax2 = fig1.add_subplot(2, 1, 2)
    l0, l1, l2, = ax2.plot(kArr, values[0,:,:], lw=2)
    s0 = ax2.scatter(params[0,1], params[0,2], c='red', s=50)
    fit, = ax2.plot(kArr, parabola(kArr, *params[0]), 'k--', lw=1)
    ax2.set_xlabel('Quasimomentum ($q/k_L$)')
    ax2.set_ylabel('Energy ($E_L$)')
    
    
    kline, = ax3.plot(tArr[0], params[0,1], color='r', lw=2)
    eline, = ax4.plot(tArr[0], params[0,2], color='r', lw=2)
    mline, = ax5.plot(tArr[0], 1/params[0,0], color='r', lw=2)
    kmax = np.max(params[:,1])
    Emin, Emax = np.min(params[:,2]), np.max(params[:,2])
    Ewidth = Emax - Emin
    mmin, mmax = np.min(1/params[:,0]), np.max(1/params[:,0])
    mwidth = mmax - mmin
    
    ax3.set_xlim(0, tMax)
    ax3.set_ylim(-1.1*kmax, 1.1*kmax)
    ax3.set_xlabel('Time (au)')
    ax3.set_ylabel('$k_min$ ($k/k_L$)')
    
    ax4.set_xlim(0, tMax)
    ax4.set_ylim(Emin - Ewidth*0.1, Emax + Ewidth*0.1)
    ax4.set_xlabel('Time (au)')
    ax4.set_ylabel('Min. Energy ($E_{min}/E_L$)')
    
    ax5.set_xlim(0, tMax)
    ax5.set_ylim(mmin - mwidth*0.1, mmax + mwidth*0.1)
    ax5.set_xlabel('Time (au)')
    ax5.set_ylabel('Eff. Mass $m^*$, (au)')
    
    
    plt.tight_layout()
    
    #Animate and save the function
    def update_lines(fnum, kArr, eigs, tArr, dArr, line0, line1, line2, s0, dline, fit):
        '''This is the animator function and updates the plot at each frame'''
        line0.set_ydata(eigs[fnum,:,0])
        line1.set_ydata(eigs[fnum,:,1])
        line2.set_ydata(eigs[fnum,:,2])
        s0.set_offsets(np.hstack((params[fnum,1, np.newaxis], 
                                  params[fnum,2, np.newaxis]))) #This is a weird line, but it works.
        fit.set_ydata(parabola(kArr, *params[fnum]))
        dline.set_data(tArr[:fnum], deltaArr[:fnum])
        kline.set_data(tArr[:fnum], params[:fnum,1])
        eline.set_data(tArr[:fnum], params[:fnum,2])
        mline.set_data(tArr[:fnum], 1/params[:fnum,0])
        return line0, line1, line2, s0,
    
    line_ani = animation.FuncAnimation(fig1, update_lines, Ntpts, 
                                       fargs=(kArr, values, tArr, deltaArr, l0, l1, l2, s0, dline, fit), 
                                       interval=100, blit=True)
    print("Saving...")
    line_ani.save('AC_dispersion_animation.mp4', writer='ffmpeg')
    print("Duration: ", time() - begtime)
    
