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
    def __init__(self, tau_param, u_param):
        self.tau = tau_param 
        self.v_vect = np.zeros(len(t_vect)) #Creates firing output vector
        self.v_vect[0] = 0 #Initial firing rate is 0 spikes/sec
        self.u_vect = u_param #Sets the input vector
        self.w_vect = inputWeightsVect_param #Sets weight vector
    def PlotSelf(self, currentInjection_param, firingRate_param):
        #Plots the current with respect to time
        currentInjection = ConstCurrent(t_vect, I_e, [t_stimStart,t_stimEnd])
        fig, axs = plt.subplots(2, constrained_layout = True)
        fig.suptitle('Box Current and Corresponding Firing Rate')
        axs[0].plot(t_vect, currentInjection)
        axs[0].set_title("Current x Time")
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("Current [nA]")

        #Plots firing rate over time using step-wise updating
        for i in range(1, len(t_vect)-1):
            firingRate_vect[i] = dt * (-firingRate_vect[i-1] + currentInjection[i-1]) * 1/tau + firingRate_vect[i-1]
        axs[1].plot(t_vect, firingRate_vect)
        axs[1].set_title("Firing Rate x Time")
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Firing rate [spikes/second]")
        plt.show()
    def InjectionCalculation(self):
        '''Calculates the firing rate over entire t_vect after an injection modelled by an input current'''
    def SynapticCalculation(self):
        '''Calculates the firing rate based on synaptic input'''
    def InputConstruction(self, inputNeuron_vect):
        '''Constructs the u vector based on all neurons that feed into it'''
    def SynapticCurrentConstruction(self):
        '''Take the inputs and calculate all the synaptic current over time'''
    def RateConstruction(self):
        '''Take the synaptic current and use the update rule and a sigmoid to update the firing rate'''
