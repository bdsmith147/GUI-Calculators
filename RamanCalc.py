import sys
from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QVBoxLayout, QApplication, QMainWindow

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import (
	FigureCanvasQTAgg as FigureCanvas,
	NavigationToolbar2QT as NavigationToolbar)
from scipy.optimize import curve_fit
from decimal import Decimal
import time

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('RamanCouplingUI.ui', self)
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout = QVBoxLayout()        
        self.plot_layout.addWidget(self.canvas)        
        #self.plot_layout.addWidget(self.toolbar)        
        self.gridLayout_2.addLayout(self.plot_layout, 1, 2, 1, 1)

        self.ax = self.figure.add_subplot(111)
        
        #Physical Constants
        self.h = 6.626e-34 #planck's constant [m^2 kg/s]
        self.hbar = self.h/(2*np.pi) #h/2pi [m^2 kg/s]
        self.m_Rb87 = 87*1.66e-27 # Mass of Rb87 atom [kg]        
        self.wavel_Rb87 = 791.1e-9 # Rb97 D2 line [m]
        self.krecoil =  2*np.pi/self.wavel_Rb87 # Recoil wavenumber [1/m]
        self.Erecoil =  (self.h/ (2*self.m_Rb87) ) * (1/self.wavel_Rb87)**2 #Recoil Energy [Hz]
        print(self.Erecoil)
        #Parameters
        self.Omega = 1
        self.epsilon = 0
        self.delta = 0
        self.kR = 2
        self.klim_max = 4
        self.klim_min = -np.copy(self.klim_max)        
        self.Npts = (self.klim_max - self.klim_min)*100 + 1
        self.karr = np.linspace(self.klim_min,self.klim_max,self.Npts)
        self.kPoint = int((self.Npts - 1)/2)
        self.OmegaSpinBox.setValue(self.Omega)
        self.deltaSpinBox.setValue(self.delta)
        self.epsilonSpinBox.setValue(self.epsilon)
        
        self.OmegaSpinBox.setSingleStep(0.5)
        self.deltaSpinBox.setSingleStep(0.5)
        self.epsilonSpinBox.setSingleStep(0.5)
        
        self.OmegaSpinBox.setRange(0,100)
        self.deltaSpinBox.setRange(-100,100)
        self.epsilonSpinBox.setRange(-100,100)
        
        self.OmegaSpinBox.valueChanged.connect(self.updateParams)
        self.deltaSpinBox.valueChanged.connect(self.updateParams)
        self.epsilonSpinBox.valueChanged.connect(self.updateParams)
        
        self.BareCheck.tristate = False
        self.CoupleCheck.tristate = False 
        self.FitMinCheck.tristate = False        
        
        self.BareCheck.setChecked(True)
        self.CoupleCheck.setChecked(True)
        self.FitMinCheck.setChecked(True)
        
        self.BareCheck.stateChanged.connect(self.updatePlot)
        self.CoupleCheck.stateChanged.connect(self.updatePlot)
        self.FitMinCheck.stateChanged.connect(self.updatePlot)
        
        self.kPointSlider.setMinimum(0)
        self.kPointSlider.setMaximum(self.Npts-1)
        self.kPointSlider.setTickInterval(1)
        self.kPointSlider.setValue(self.kPoint)
        self.kPointSlider.valueChanged.connect(self.moveSlider)
        
        self.kPointLE.editingFinished.connect(self.updateSliderText)    
        self.ResetButton.clicked.connect(self.reset_kPoint)
    
        self.updateParams()
        self.set_kPoint(self.kPoint)
        self.show()
        
    def calcEigen(self):       
        evalues = []
        evectors = []
        uncoupled = []
        for k in self.karr:
            H = np.array([[(k+self.kR)**2 - self.delta, self.Omega/2, 0],
                           [self.Omega/2, k**2-self.epsilon, self.Omega/2],
                            [0, self.Omega/2, (k-self.kR)**2 + self.delta]])
            evals, evecs = np.linalg.eig(H)
            evecs = evecs
            idx = evals.argsort()[::-1]
            evals = evals[idx]
            evecs = evecs[:,idx]
#            if self.Omega != 0.0:
#                #Sorts the eigenvalues into non-insersecting dispersion bands                
#                elem = np.argsort(evals)
#                evals = np.take(evals, elem)
#                evecs = np.take(evecs, elem, axis=0)
            evalues.append(evals)
            evectors.append(evecs)
            uncoupled.append(np.diag(H)) #only takes the diagonal entries of the matrix, i.e. "uncoupled"
        self.values = np.array(evalues) # Creates an array of three subarrays corresponding to each of the eigenstates' dispersions
        self.vectors = np.absolute(np.array(evectors))
        self.uncoupled = np.array(uncoupled)
        
#        plt.figure()
#        plt.plot(self.karr, self.vectors[:,:,0]**2)
#        plt.show()
#        plt.figure()
#        plt.plot(self.karr, self.vectors[:,:,2]**2)
#        plt.show()
        

    def updatePlot(self):
        self.ax.clear()
        if self.BareCheck.isChecked():
            self.ax.plot(self.karr, self.uncoupled, 'gray', linestyle='dashed', lw=2)
        if self.CoupleCheck.isChecked():
            self.ax.plot(self.karr, self.values, lw=2)
            self.kpt_marker = self.ax.scatter(self.karr[self.kPoint], self.val1[self.kPoint], marker='o', s=100, facecolors='none', linewidths=2, c='orange')
        if self.FitMinCheck.isChecked():
            self.ax.plot(self.karr, self.fitted, 'r-.', lw=2)
            self.ax.scatter(self.kMin, self.Emin, zorder=50, marker='o', s=100, facecolors='none', linewidths=2)     
        self.ax.set_xlabel('Quasimomentum [k/$k_R$]')
        self.ax.set_ylabel('Energy [E/$E_R$]')
        self.canvas.draw()
    
    def updateMarker(self):
        self.kpt_marker.remove()
        self.kpt_marker = self.ax.scatter(self.karr[self.kPoint], self.val1[self.kPoint], marker='o', s=100, facecolors='none', linewidths=2, c='orange')
        self.canvas.draw()
        
    def updateParams(self):
        self.Omega = self.OmegaSpinBox.value()
        self.delta = self.deltaSpinBox.value()
        self.epsilon = self.epsilonSpinBox.value()        
        self.calcEigen()
        self.fitMin()
        self.set_kPoint(self.kPoint)
        self.updatePlot()

    def fitMin(self):
        self.val1 = self.values[:,2] #The ground eigenstate dispersion
        fitwindow = 100 #Don't fit the entire dispersion to a parabola, just the area inside the window
        roughmin = np.argmin(self.val1) #Initial guess for the the minimum of the parabola
        guesses = [self.krecoil, self.karr[roughmin], self.val1[roughmin]]
        kWindow = self.karr[roughmin-fitwindow:roughmin+fitwindow]
        valWindow = self.val1[roughmin-fitwindow:roughmin+fitwindow]
        fit, pcov = curve_fit(self.parabola, kWindow, valWindow, guesses)
        self.kMin, self.Emin = fit[1:3]
        self.fitted = self.parabola(self.karr, fit[0], fit[1], fit[2])

    def set_kPoint(self, point):
        self.kPoint = point #kPoint is an number corresponding to an index of self.karr
        self.pop1, self.pop2, self.pop3 = self.vectors[self.kPoint,:,2]**2
        self.pop1Val.setText('%.3f'%(self.pop1))
        self.pop2Val.setText('%.3f'%(self.pop2))
        self.pop3Val.setText('%.3f'%(self.pop3))
        self.kPointLE.setText('%.3f'%self.karr[point])
        #Python can't seem to accurately represent and print certain values.
        #That's why entering a 2.0 will be printed as a 1.990, for example.
        #This following line prints the number as it is represented in binary.
        #print Decimal(self.karr[point]) 
        
    def reset_kPoint(self):
        kMin_elem = self.kFloatToInt(self.kMin)
        self.set_kPoint(kMin_elem)
        self.kPointSlider.setSliderPosition(kMin_elem)
        self.updateMarker()
    
    def moveSlider(self):
        newPoint = self.kPointSlider.sliderPosition()     
        self.set_kPoint(newPoint)
        self.updateMarker()

    def updateSliderText(self):
        thisval = float(self.kPointLE.text())
        tar = self.kFloatToInt(thisval)
        self.kPointSlider.setSliderPosition(tar)
        self.updateMarker()        
        
    def kFloatToInt(self, target):
        target_karr = self.karr - float(target) 
        #Converts the minimum (float) to an array index (int)
        target_elem = np.where(np.diff(np.signbit(target_karr)))[0][0]      
        return target_elem
    
    def parabola(self, x, A, x0, y0):
        return A**2 * (x - x0)**2 + y0
        
if __name__ == '__main__':
    app = 0    
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())