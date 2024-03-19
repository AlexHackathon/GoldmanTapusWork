import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression

dt = 0.1 #Time step [ms]
t_stimStart = 100 #Time start box current [ms]
t_stimEnd = 500 #Time stop box current [ms]
t_end = 1000 #Time to stop the simulation [ms]
I_e = 1.43 #Magnitude of current

tau = 20 #membrane time constant [ms]
t_vect = np.arange(0, t_end, dt) #Creates time vector[ms] with time step dt[ms]

#Creates a box current over the time vector with magnitude stimMag
def ConstCurrent(time_vect, stimMag, stimStartEnd_vect):
    addCurrent = False
    i = 0
    current_vect = np.zeros(len(time_vect))
    for x in range(0, len(time_vect)-1):
        if i >= len(stimStartEnd_vect):
            continue
        elif time_vect[x] >= stimStartEnd_vect[i]:
            addCurrent = not addCurrent
            i = i + 1
        if addCurrent:
            current_vect[x] = stimMag
    return current_vect

def Activation1(a_param, p_param, r_param):
    numerator = r_param ** p_param
    denominator = a_param + r_param ** p_param
    return numerator/denominator

def Activation2(a_param, p_param, r_param):
    numerator = r_param
    denominator = a_param - r_param
    return numerator/denominator


current1_vect = ConstCurrent(t_vect, 30, [t_stimStart, t_stimEnd]) #Current injected into the neuron [spikes/sec]
weightMatrix = np.array([.5])
v1a_vect = np.zeros(len(t_vect))
tIdx = 1

#Linear version
while tIdx < len(t_vect):
    totalInput = np.dot([v1a_vect[tIdx-1]], weightMatrix) + current1_vect[tIdx-1]
    v1a_vect[tIdx] = v1a_vect[tIdx-1] + dt/tau*(-v1a_vect[tIdx-1] + totalInput)
    tIdx = tIdx + 1

plt.plot(t_vect, v1a_vect)
plt.plot(t_stimStart + tau, .63 * 30, 'ro')
plt.show()
#***********************************************************************************
#Begin linear regression fitting
timeCalcStart = 550 #Time to start fitting from [ms]
trainTIdx = np.searchsorted(t_vect, timeCalcStart)
#trainTIdxEnd = np.searchsorted(t_vect, timeCalcEnd)
t_cut = t_vect[trainTIdx:] - t_vect[trainTIdx]
for i in range(0,10):
    print(t_cut[i])
lnR_cut = np.log(v1a_vect[trainTIdx:])
t_cut = t_cut.reshape(-1,1)
lnR_cut = lnR_cut.reshape(-1,1)
plt.scatter(t_cut, lnR_cut)
plt.show()
reg = LinearRegression()
reg.fit(t_cut,lnR_cut)
print(reg.coef_)
print(reg.intercept_)
print("Tau is: " + str(-1/reg.coef_))
print("R0 is: " + str(math.exp(reg.intercept_[0])))
print("R0 is actually: " + str(v1a_vect[trainTIdx]))
