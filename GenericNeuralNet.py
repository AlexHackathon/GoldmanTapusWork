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

class Neuron:
    def __init__(self, current_param, ):
        self.current_vect = current_param
        self.v_vect = np.zeros(len(t_vect))
    def UpdateRule(self, weights_mat, tIdx_param, u_vect, dt_param, tau_param):
        totalInput = -self.v_vect[tIdx_param-1] + self.current_vect[tIdx_param-1] + sum(np.dot(u_vect, weights_mat))
        self.v_vect[tIdx_param] = self.v_vect[tIdx_param-1] + dt_param/tau_param*totalInput
    def UpdateRuleNonlineardef (self, weights_mat, tIdx_param, u_vect, dt_param, tau_param, activationFunction):
        totalInput = -self.v_vect[tIdx_param-1] + self.current_vect[tIdx_param-1] + activationFunction(sum(np.dot(u_vect, weights_mat)))
        self.v_vect[tIdx_param] = self.v_vect[tIdx_param-1] + dt_param/tau_param*totalInput
        
weightMatrix = np.array([[0,1],
                        [1,0]])
tIdx = 1
neuronPop = [] #Array of neurons in the same order as they appear in the weights

while tIdx < len(t_vect):
    for n in neuronPop:
        u_vect_temp = np.array([])
        for n in neuronPop:
            u_vect_temp.append(n.v_vect[tIdx-1])
        n.UpdateRule(weightMatrix, tIdx, u_vect_temp, dt, tau)
    tIdx = tIdx + 1
