import numpy as np
import matplotlib.pyplot as plt
from po2code import *

a = np.load('LSD_skintraining_highmelanin_invitro_flowpredictions.npy')
b = np.load('LSTM_skintraining_highmelanin_invitro_flowpredictions.npy')
c = np.load('LU_invitro_flowpredictions.npy')

size_of_timestep = 1417/86
times = [size_of_timestep*i for i in range(87)]
d = load_po2(['I:/research/seblab/data/group_folders/Tom/Melanin_Flow Phantom/20211202 - Run 1 (high melanin)/pO2data.mat'])

po2_times =d.loc[:,'Time']
po2_times = np.array(po2_times/10**9)
so2_vals = d.loc[:,'so2 (Post)']
so2_vals = np.array(so2_vals/100)

plt.plot(po2_times,so2_vals,label='pO2')
plt.plot(times,a, label = 'LSD')
plt.plot(times,b, label = 'SASD')
plt.plot(times,c, label = 'LU')
plt.xlabel('Time (s)')
plt.ylabel('sO2')
plt.legend(loc = 'lower left')
plt.show()
