import os
import dva_noisesrc_gain_correction
import importlib
import numpy as np

#phase = 2
#days = [2,3,4,5,6,7,8,9,10]
#days = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
#Ne =   [2,2,2,2,1,2,2,2,2, 2, 2, 1, 2, 1, 2, 2, 2]
#Nm =   [2,2,2,2,1,2,2,2,2, 2, 2, 1, 2, 1, 2, 2, 2]
#days = np.concatenate([np.arange(25,30,1),np.arange(31,41,1),[43,44,45]])
#print(days)
#print(len(days))
#Ne = np.concatenate([[1]*11,[2]*2,[1]*5])
#Nm = np.concatenate([[1]*11,[2]*2,[1]*5])
#print(Nm)
#print(len(Nm))

#phase = 1
#days = np.arange(2,48,1)
#print(len(days))
#Ne = np.concatenate([[1]*31,[2]*12,[1],[2]*2])
#Nm = np.concatenate([[1]*31,[2]*12,[1],[2]*2])
#print(len(Ne))                    
#days = [1]
#Ne = [0]
#Nm = [1]

phase = 3
#days = np.concatenate([np.arange(1,9,1),np.arange(10,56,1)])
#days = np.concatenate([np.arange(5,9,1),np.arange(10,56,1)])
#days = [59,60,61,62,63]
days = [58]
print(days)
print(len(days))
#Ne = [1,1,1,1,1]
#Nm = [0,0,1,0,0]
#Ne = [1]*50
Ne = [1]
#Nm = np.concatenate([[1,0,1,0,0],[1]*16,[0],[1]*25,[0],[1]*6])
#Nm = np.concatenate([[0],[1]*16,[0],[1]*25,[0],[1]*6])
Nm = [0]
print(Ne)
print(len(Ne))
print(Nm)
print(len(Nm))

importlib.reload(dva_noisesrc_gain_correction)

for i in range(0,len(days)):
    
    dva_noisesrc_gain_correction.noise_source_correct(phase,days[i],Ne[i],Nm[i])
    
