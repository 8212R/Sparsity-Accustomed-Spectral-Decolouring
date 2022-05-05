import numpy as np
import matplotlib.pyplot as plt
from po2code import *

a = np.load('LSD_flowtraining_nomelanin_2_invitro_flowpredictions.npy')
a_err = np.load('LSD_stddev.npy')
b = np.load('LSTM_flowtraining_nomelanin_2_invitro_flowpredictions.npy')
b_err = np.load('LSTM_stddev.npy')
c = np.load('LU_invitro_flowpredictions.npy')
c_err = np.load('LU_stddev.npy')
a_clean = a[:-1].copy()
b_clean = b[:-1].copy()
c_clean = c[:-1].copy()
a_err_clean = a_err[:-1].copy()
b_err_clean = b_err[:-1].copy()
c_err_clean = c_err[:-1].copy()

size_of_timestep =3704.5/231
times = np.array([size_of_timestep*i for i in range(231)])

part1 = load_po2(['I:/research/seblab/data/group_folders/Tom/Melanin_Flow Phantom/20211125 - Run 0 (no melanin)/pO2 Probe Data/pO2data_1a.mat'])
po2_times_1 =part1.loc[:,'Time']
po2_times_1 = np.array(po2_times_1/10**9)
so2_vals_1 = part1.loc[:,'so2 (Pre)']
so2_vals_1 = np.array(so2_vals_1/100)

part2 = load_po2(['I:/research/seblab/data/group_folders/Tom/Melanin_Flow Phantom/20211125 - Run 0 (no melanin)/pO2 Probe Data/pO2data_1b.mat'])
po2_times_2 =part2.loc[:,'Time']
po2_times_2 = np.array(po2_times_2/10**9) + max(po2_times_1)
so2_vals_2 = part2.loc[:,'so2 (Pre)']
so2_vals_2 = np.array(so2_vals_2/100)

so2_vals = np.concatenate((so2_vals_1, so2_vals_2))
po2_times = np.concatenate((po2_times_1,po2_times_2))
so2_vals_truncated = np.array([so2_vals[i] if po2_times[i] < 2520 else np.NaN for i in range(len(so2_vals))])

#plt.plot(po2_times,so2_vals*100,label='pO2')
plt.plot(po2_times[1866:]-po2_times[1866],so2_vals_truncated[1866:]*100,label = 'pO$_2$')
plt.plot(times[75:]-times[75],a_clean[75:]*100, label = 'LSD', color='darkorange')
plt.fill_between(times[75:]-times[75],(a_clean-a_err_clean)[75:]*100,(a_clean+a_err_clean)[75:]*100, alpha = 0.5, color = 'darkorange', linewidth = 0)
plt.plot(times[75:]-times[75],b_clean[75:]*100, label = 'SASD', color = 'g')
plt.fill_between(times[75:]-times[75],(b_clean-b_err_clean)[75:]*100,(b_clean+b_err_clean)[75:]*100, alpha = 0.5, color = 'g', linewidth = 0)
plt.plot(times[75:]-times[75],c_clean[75:]*100, label = 'LU', color = 'r')
plt.fill_between(times[75:]-times[75],(c_clean-c_err_clean)[75:]*100,(c_clean+c_err_clean)[75:]*100, alpha = 0.5, color = 'r', linewidth = 0)
plt.xlabel('Time (s)')
plt.ylabel('sO$_2$ [%]')
plt.legend(loc = 'upper right')
#plt.savefig('Figure13b.png')
plt.show()
print(len(times))
print(len(po2_times))