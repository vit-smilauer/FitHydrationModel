#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 30, 2025

@author: smilauer
"""
#Fitting affinity and exponential hdyration model to isothermal calorimetry data

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import pandas as pd
import re


#Extract three digits from input file name
try:
    idNum = int(sys.argv[1][0:3])
    #print(idNum)
except:
    print("The first argument is the filename starting with three digits and containing measured data from isothermal calorimetry")
    sys.exit(0)


B1r = (0.1, 3)
B2r = (1.e-8, 1.e-2)
etar = (4.,14.)
DoHInfr = (0.85 -1.e-4, 0.85 +1.e-4)
DoH1r = (10.0, 10.4) #unused
P1r = (1.e-2, 5.) #unused

if idNum in [100,103,105,106,109,114,115,116,122,158,163,172,179,202]: #CEM I
    Q = 500. #J/g, potential heat
    activationEnergy = 38.3e+3
elif idNum in [107,108,131,135,146,147,148,165,180,186,188,193,195]: #slag 15% CEM II/A-S, CEM II/A-LL, CEM II/A-M ,143 vyhozen, není realistický
    Q = 460. #J/g
    activationEnergy = 40.0e+3
elif idNum in [196,197]:
    Q = 460. #J/g
    activationEnergy = 40.0e+3
    B1r = (0.5, 1.0)
elif idNum in [101,102,110,111,118,132,134,136,137,149,151,152,153,155,156,159,160,161,167,173,181,182,187,189,198,200]: #slag 25-34% CEM II/B-S, CEM II/B-LL, CEM II/B-M, CEM V/A, 144 vyhozen
    Q = 420. #J/g
    activationEnergy = 45.0e+3
elif idNum in [183]: #slag 35-49%, CEM II/C-M
    Q = 400. #J/g
    activationEnergy = 47.0e+3
elif idNum in [112,119,121,157,184,199,201]: #slag 50-70%, CEM III/A, CEM III/B
    Q = 380. #J/g
    activationEnergy = 48.0e+3
elif idNum in [162]: #SORFIX
    Q = 400. #J/g
    activationEnergy = 45.0e+3
elif idNum in [174]: #H-cement
    Q = 400. #J/g
    activationEnergy = 45.0e+3
    B1r = (0.1, 2.0)
else:
    print('Number %d not in the list' % idNum)
    sys.exit(0)


#Read input file
myLines = []
with open(sys.argv[1]) as f:
    print('Opening file: ', sys.argv[1])
    for line in f:
        if re.match(r"^([-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?) ([-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?) ([-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?)$", line): #find three numbers separated by space 
            list1=list(line.split()) #split string to numbers
            list2=list(map(float,list1)) #convert from strings to floats
            #print(list2)
            myLines.append(list2) #create a list of lists
            
#print(myLines)
df = pd.DataFrame(myLines, columns=['Time', 'Flow', 'Heat'])
pd.options.display.float_format = '{:.10f}'.format
lowestTime = df['Time'].iloc[0]
highestTime = df['Time'].iloc[-1]
print("Lowest and highest times (h):", lowestTime, highestTime)
#exit(0)

#Parameters of hydration. CEM I 42,5 R sc Mokrá
#B1=1.2667 #h^-1
#B2=8.e-6
#eta=7.4
#DoHInf=0.75


NumberOfTimePoints = 300 #should be around 300 to have good precision
#set up points of interest linearly on log scale (geometrical series), including end points
discTimes = np.logspace(math.log10(lowestTime), math.log10(highestTime), num=NumberOfTimePoints)
#print("Time points [h]", discTimes)

#Interpolate values from dataframe for discTime
discExperQ = np.interp(discTimes, df['Time'], df['Heat'])

#print("Experimentally measured released heat [J/g]", discExperQ)


#Use a class so we can store DoHMax etc. easily
class AffModel:
    def __init__(self):
        self.DoHMax = 0.
        self.TimeMax = 0.

    def getQ(self,discTimes, B1, B2, eta, DoHInf, DoH1, P1, oneReturn=True):
        #print('B1 %f B2 %f eta %f DoHInf %f DoH1 %f P1 %f' % ( B1, B2, eta, DoHInf, DoH1, P1))
        DoH = 0.
        Time = discTimes[0]
        self.TimeMax = discTimes[-1]
        discDoH=np.zeros(len(discTimes))
        discFlow=np.zeros(len(discTimes))
        j=0
        for j in range(1,len(discTimes)):#skip the first (0th) element
            #print(j, discTimes[j])
            dTime = discTimes[j]-discTimes[j-1]
            DoH_old = DoH + self.scaleWithTemperature(20.) * self.affinity25(DoH, B1, B2, eta, DoHInf, DoH1, P1) * dTime #predictor
            #http://en.wikipedia.org/wiki/Predictor%E2%80%93corrector_method
            #corrector - integration through trapezoidal rule
            #3 loops normally suffices                               
            for i in range(4):
                aff = self.scaleWithTemperature(20.) * ( self.affinity25(DoH, B1, B2, eta, DoHInf, DoH1, P1) + self.affinity25(DoH_old, B1, B2, eta, DoHInf, DoH1, P1) ) / 2.;
                DoH_new = DoH + aff*dTime
                DoH_old = DoH_new
            discFlow[j] = aff*Q
            DoH = DoH_new
            self.DoHMax = DoH
            discDoH[j] = DoH
            #print("Time:", discTimes[j], discDoH[j], discFlow[j])
        if(oneReturn):
            return Q*discDoH
        else:
            return (Q*discDoH, discFlow)

    
    def affinity25 (self,DoH, B1, B2, eta, DoHInf, DoH1, P1):
        fs=1.
        if DoH>DoH1:
            fs=1+P1*(DoH-DoH1)
        result = B1 * ( B2 / DoHInf + DoH ) * ( DoHInf - DoH ) * fs * math.exp(-eta * DoH / DoHInf);
        if (result < 0.):
            return 0.
        else:
            return result

    def scaleWithTemperature (self,T):
        return math.exp( activationEnergy / 8.314 * ( 1. / ( 273.15 + 25 ) - 1. / ( 273.15 + T ) ) ); 


class ExpModel:
    def __init__(self):
        pass
    def getQ(self, discTimes, tau, beta, DoHInf, oneReturn=True):
        ret = Q*DoHInf*np.exp(-(tau/discTimes)**beta)
        if(oneReturn):
            return ret
        else:
            retFlow = Q*DoHInf/discTimes*beta*(tau/discTimes)**beta * np.exp(-(tau/discTimes)**beta)
            return (ret, retFlow)
    

#discQ = getQ(discTimes, 1.2667, 8.e-6, 7.4, 0.85)
#print(discQ)

#weights - not much useful
sigma = np.ones(len(discTimes))
sigma[discExperQ > 260] = 1.
#print(sigma)

A = AffModel()
B = ExpModel()

params, cov = curve_fit(A.getQ, discTimes, discExperQ, method='trf', bounds=( [B1r[0], B2r[0], etar[0], DoHInfr[0], DoH1r[0], P1r[0]], [B1r[1], B2r[1], etar[1], DoHInfr[1] , DoH1r[1], P1r[1]]), sigma=sigma, absolute_sigma=False)
print('Fitted parameters affinity:', params[0:4], Q, activationEnergy, 'DoH_max:', '{:1.4f}'.format(A.DoHMax), 'Time_max:', '{:1.2f}'.format(A.TimeMax) )

params1, cov1 = curve_fit(B.getQ, discTimes, discExperQ, method='trf', bounds=( [0.1,0.1, DoHInfr[0]], [1000, 5, DoHInfr[1]] ), sigma=sigma, absolute_sigma=False)
print('Fitted parameters exponential:', params1, Q )


discAffQ = A.getQ(discTimes, params[0], params[1], params[2], params[3], params[4], params[5])
discQExponQ = B.getQ(discTimes, params1[0], params1[1], params1[2])

if 0:
    fig, ax = plt.subplots()
    ax.plot(discTimes, discExperQ, '.-', label='Experiment')
    ax.plot(discTimes, discAffQ, '.-', label='Affinity')
    ax.plot(discTimes, discQExponQ , '.-', label='Exponential')
    plt.legend(loc="lower right")
    ax.set_xscale('log')
    ax.set_xlabel('Hydration time at 20°C (h)')
    ax.set_ylabel('Released heat (J/g)')

#Calculate coefficient of variation
#Get standard error, differece of two vectors
diff = discQExponQ - discExperQ #eponential model
#diff = discAffQ - discExperQ #affinity model
mean = np.mean(discExperQ)
diff = diff * diff
sumDiff = np.sum(diff)


#print('AAA',sys.argv[1][0:3], sumDiff, len(discAffQ), mean, np.mean(discTimes))

#Extend data series to one year
factor = discTimes[-1]/discTimes[-2]
while (discTimes[-1] < 365*24):
    discTimes = np.append(discTimes, [discTimes[-1]*factor])
    #print(discTimes[-1])

discAffQ, discAffFlow = A.getQ(discTimes, params[0], params[1], params[2], params[3], params[4], params[5], oneReturn=False)
results = np.column_stack((discTimes, discAffQ/Q, discAffQ, discAffFlow/3600*1000)) #align vectors next to each other
#print(results)
np.savetxt(sys.argv[1][0:3]+'-Fit-aff.csv', results, delimiter=" ", header='Fitted affinity hydration model, B1 %f h^-1, B2 %1.3e, eta %0.4f, DoHInf %0.2f, Qpot %0.1f J/g, Ea %0.1f J/mol\nTime(h) DoH Q(J/g) q(mW/g)' % (params[0], params[1], params[2], params[3], Q, activationEnergy), fmt='%.4f %.2e %.4f %.5f')

discExpQ, discExpFlow = B.getQ(discTimes, params1[0], params1[1], params1[2], oneReturn=False)
results = np.column_stack((discTimes, discExpQ/Q, discExpQ, discExpFlow/3600*1000)) #align vectors next to each other
#print(results)
np.savetxt(sys.argv[1][0:3]+'-Fit-exp.csv', results, delimiter=" ", header='Fitted exponential hydration model, tau %f h, beta %f, DoHInf %0.2f, Qpot %0.1f J/g, Ea %0.1f J/mol\nTime(h) DoH Q(J/g) q(mW/g)' % (params1[0], params1[1], params1[2], Q, activationEnergy), fmt='%.4f %.2e %.4f %.5f')


plt.show()

