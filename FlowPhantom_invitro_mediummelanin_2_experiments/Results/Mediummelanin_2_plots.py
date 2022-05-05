import numpy as np
import matplotlib.pyplot as plt
from po2code import *

a = np.load('LSD_skintraining_mediummelanin_2_invitro_flowpredictions.npy')
a_err = np.load('LSD_stddev.npy')
b = np.load('LSTM_skintraining_mediummelanin_2_invitro_flowpredictions.npy')
b_err = np.load('LSTM_stddev.npy')
c = np.load('LU_invitro_flowpredictions.npy')
c_err = np.load('LU_stddev.npy')

size_of_timestep = 4637.835/283
times = np.array([size_of_timestep*i for i in range(284)])
d = load_po2(['I:/research/seblab/data/group_folders/Tom/Melanin_Flow Phantom/20211202 - Run 3 (medium melanin_2)/pO2data.mat'])

po2_times =d.loc[:,'Time']
po2_times = np.array(po2_times/10**9)
so2_vals = d.loc[:,'so2 (Post)']
so2_vals = np.array(so2_vals/100)

plt.plot(po2_times[2507:]-po2_times[2507],so2_vals[2507:]*100,label='pO$_2$')
plt.plot(times[104:]-times[104],a[104:]*100, label = 'LSD', color='darkorange')
plt.fill_between(times[104:]-times[104],(a-a_err)[104:]*100,(a+a_err)[104:]*100, alpha = 0.5, color = 'darkorange', linewidth = 0)
plt.plot(times[104:]-times[104],b[104:]*100, label = 'SASD', color = 'g')
plt.fill_between(times[104:]-times[104],(b-b_err)[104:]*100,(b+b_err)[104:]*100, alpha = 0.5, color = 'g', linewidth = 0)
plt.plot(times[104:]-times[104],c[104:]*100, label = 'LU', color = 'r')
plt.fill_between(times[104:]-times[104],(c-c_err)[104:]*100,(c+c_err)[104:]*100, alpha = 0.5, color = 'r', linewidth = 0)
plt.xlabel('Time (s)')
plt.ylabel('sO$_2$ [%]')
plt.legend(loc = 'upper right')
#plt.savefig('Figure13a.png')
plt.show()
print(len(po2_times))
print(len(times))