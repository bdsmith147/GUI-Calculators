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
#import time

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('RFCouplingUI.ui', self)
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
        self.wavel_Rb87 = 780.1e-9 # Rb97 D2 line [m]
        self.krecoil =  2*np.pi/self.wavel_Rb87 # Recoil wavenumber [1/m]
        self.Erecoil =  (self.h/ (2*self.m_Rb87) ) * (1/self.wavel_Rb87)**2 #Recoil Energy [Hz]
        
        #Parameters
        self.Omega = 2
        self.epsilon = 0
        self.delta_lim_max = 4
        self.delta_lim_min = -np.copy(self.delta_lim_max)        
        self.Npts = (self.delta_lim_max - self.delta_lim_min)*100 + 1
        self.deltaArr = np.linspace(self.delta_lim_min,self.delta_lim_max,self.Npts)
        self.deltaPoint = int((self.Npts - 1)/2)
        
        self.OmegaSpinBox.setValue(self.Omega)
        self.epsilonSpinBox.setValue(self.epsilon)
        
        self.OmegaSpinBox.setSingleStep(0.5)
        self.epsilonSpinBox.setSingleStep(0.5)
        
        self.OmegaSpinBox.setRange(0,100)
        self.epsilonSpinBox.setRange(-100,100)
        
        self.OmegaSpinBox.valueChanged.connect(self.updateParams)
        self.epsilonSpinBox.valueChanged.connect(self.updateParams)
        
        self.BareCheck.tristate = False
        self.CoupleCheck.tristate = False      
        
        self.BareCheck.setChecked(True)
        self.CoupleCheck.setChecked(True)
        
        self.BareCheck.stateChanged.connect(self.updatePlot)
        self.CoupleCheck.stateChanged.connect(self.updatePlot)
        
        self.deltaPointSlider.setMinimum(0)
        self.deltaPointSlider.setMaximum(self.Npts-1)
        self.deltaPointSlider.setTickInterval(1)
        self.deltaPointSlider.setValue(self.deltaPoint)
        self.deltaPointSlider.valueChanged.connect(self.moveSlider)
        
        self.deltaPointLE.editingFinished.connect(self.updateSliderText)    
        self.ResetButton.clicked.connect(self.reset_kPoint)
        
        self.updateParams()
        self.set_deltaPoint(self.deltaPoint)
        self.show()
        
    def calcEigen(self):       
        evalues = []
        evectors = []
        uncoupled = []
        for delta in self.deltaArr:
            H = np.array([[delta, self.Omega/2, 0],
                           [self.Omega/2, self.epsilon, self.Omega/2],
                            [0, self.Omega/2, -delta]])
            evals, evecs = np.linalg.eig(H)
            if self.Omega != 0.0:
                #Sorts the eigenvalues into non-insersecting dispersion bands                
                elem = np.argsort(evals)
                evals = np.take(evals, elem)
                evecs = np.take(evecs, elem, axis=0)
            evalues.append(evals)
            evectors.append(evecs)
            uncoupled.append(np.diag(H)) #only takes the diagonal entries of the matrix, i.e. "uncoupled"
        self.values = np.array(evalues) # Creates an array of three subarrays corresponding to each of the eigenstates' dispersions
        self.vectors = np.absolute(np.array(evectors))
        self.uncoupled = np.array(uncoupled)
        self.val1 = self.values.T[0] #The ground eigenstate dispersion

    def updatePlot(self):
        self.ax.clear()
        if self.BareCheck.isChecked():
            self.ax.plot(self.deltaArr, self.uncoupled, 'gray', linestyle='dashed', lw=2)
        if self.CoupleCheck.isChecked():
            self.ax.plot(self.deltaArr, self.values, lw=2)
            self.delta_pt_marker = self.ax.scatter(self.deltaArr[self.deltaPoint], self.val1[self.deltaPoint], marker='o', s=100, facecolors='none', linewidths=2, c='orange')
        
#        if self.FitMinCheck.isChecked():
#            self.ax.plot(self.deltaArr, self.fitted, 'r-.', lw=2)
#            self.ax.scatter(self.kMin, self.Emin, zorder=50, marker='o', s=100, facecolors='none', linewidths=2)     
        self.ax.set_xlabel('Detuning [$\delta$]')
        self.ax.set_ylabel('Energy [E/$E_R$]')
        self.canvas.draw()
    
    def updateMarker(self):
        self.delta_pt_marker.remove()
        self.delta_pt_marker = self.ax.scatter(self.deltaArr[self.deltaPoint], self.val1[self.deltaPoint], marker='o', s=100, facecolors='none', linewidths=2, c='orange')
        self.canvas.draw()
        
    def updateParams(self):
        self.Omega = self.OmegaSpinBox.value()
        self.epsilon = self.epsilonSpinBox.value()        
        self.calcEigen()
        #self.fitMin()
        self.set_deltaPoint(self.deltaPoint)
        self.updatePlot()

#    def fitMin(self):
#        self.val1 = self.values.T[0] #The ground eigenstate dispersion
#        fitwindow = 100 #Don't fit the entire dispersion to a parabola, just the area inside the window
#        roughmin = np.argmin(self.val1) #Initial guess for the the minimum of the parabola
#        guesses = [self.krecoil, self.karr[roughmin], self.val1[roughmin]]
#        kWindow = self.karr[roughmin-fitwindow:roughmin+fitwindow]
#        valWindow = self.val1[roughmin-fitwindow:roughmin+fitwindow]
#        fit, pcov = curve_fit(self.parabola, kWindow, valWindow, guesses)
#        self.kMin, self.Emin = fit[1:3]
#        self.fitted = self.parabola(self.karr, fit[0], fit[1], fit[2])

    def set_deltaPoint(self, point):
        self.deltaPoint = point #kPoint is an number corresponding to an index of self.karr
        self.pop1, self.pop2, self.pop3 = self.vectors[self.deltaPoint,0]**2
        self.pop1Val.setText('%.3f'%(self.pop1))
        self.pop2Val.setText('%.3f'%(self.pop2))
        self.pop3Val.setText('%.3f'%(self.pop3))
        self.deltaPointLE.setText('%.3f'%self.deltaArr[point])
        #Python can't seem to accurately represent and print certain values.
        #That's why entering a 2.0 will be printed as a 1.990, for example.
        #This following line prints the number as it is represented in binary.
        #print Decimal(self.karr[point]) 
        
    def reset_kPoint(self):
        del0 = (self.Npts - 1)/2
        self.set_deltaPoint(del0)
        self.deltaPointSlider.setSliderPosition(del0)
        self.updateMarker()
    
    def moveSlider(self):
        newPoint = self.deltaPointSlider.sliderPosition()     
        self.set_deltaPoint(newPoint)
        self.updateMarker()

    def updateSliderText(self):
        thisval = float(self.deltaPointLE.text())
        tar = self.deltaFloatToInt(thisval)
        self.deltaPointSlider.setSliderPosition(tar)
        self.updateMarker()        
        
    def deltaFloatToInt(self, target):
        target_delArr = self.karr - float(target) 
        #Converts the minimum (float) to an array index (int)
        target_elem = np.where(np.diff(np.signbit(target_delArr)))[0][0]      
        return target_elem
    
#    def parabola(self, x, A, x0, y0):
#        return A**2 * (x - x0)**2 + y0
        
if __name__ == '__main__':
    app = 0    
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())