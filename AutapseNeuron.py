import numpy as np
import matplotlib.pyplot as plt
import math

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


current1_vect = ConstCurrent(t_vect, 30, [t_stimStart, t_stimEnd]) #Current injected into the neuron [spikes/sec]
weightMatrix = np.array([-.5])
v1a_vect = np.zeros(len(t_vect))
tIdx = 1

while tIdx < len(t_vect):
    totalInput = -v1a_vect[tIdx-1] + np.dot([v1a_vect[tIdx-1]], weightMatrix) + current1_vect[tIdx-1]
    v1a_vect[tIdx] = v1a_vect[tIdx-1] + dt/tau*totalInput
    tIdx = tIdx + 1

plt.plot(t_vect, v1a_vect)
#plt.plot(t_stimStart + tau, .63 * 30, 'ro')
plt.show()
