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


current1a_vect = ConstCurrent(t_vect, 30, [t_stimStart, t_stimEnd]) #Current injected into Neuron 1 [spikes/sec]
current1b_vect = ConstCurrent(t_vect, 40, [t_stimStart, t_stimEnd]) #Current injected into Neuron 2 [spikes/sec]
weightMatrix = np.array([[0,0],
                        [0,0]]) #Adjustable weight matrix (-1 to 1 preferrable)
v1a_vect = np.zeros(len(t_vect)) #Output of Neuron 1 [spikes/sec]
v1b_vect = np.zeros(len(t_vect)) #Output of Neuron 2 [spikes/sec]
tIdx = 1 #Time index used for iteration

while tIdx < len(t_vect):
    #Calculating current in Neuron 1
    u1a_vect = [0,v1b_vect[tIdx-1]] #Input of each neuron
    totalInput1a = -v1a_vect[tIdx-1] + current1a_vect[tIdx-1] + sum(np.dot(u1a_vect, weightMatrix)) #Total input including decay and injected current
    v1a_vect[tIdx] = v1a_vect[tIdx-1] + dt/tau*totalInput1a #Stepwise calculation of the next firing rate

    #Calculating current in Neuron 2
    #See previous comment
    u1b_vect = [v1a_vect[tIdx-1],0] 
    totalInput1b = -v1b_vect[tIdx-1] + current1b_vect[tIdx-1] + sum(np.dot(u1b_vect, weightMatrix))
    v1b_vect[tIdx] = v1b_vect[tIdx-1] + dt/tau*totalInput1b

    #Increase the time index
    tIdx = tIdx + 1

plt.plot(t_vect, v1a_vect)
plt.show()
plt.plot(t_vect, v1b_vect)
plt.show()
