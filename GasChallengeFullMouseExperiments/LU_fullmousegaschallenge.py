# Script performs an approximate linear unmixing method (using inversion) to estimate sO2
import torch
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.optimize import minimize
import itertools
import h5py
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('rb',['b','w','r'], N=256)

#######################################################################################################################

# Altering tabulated data to match the wavelengths used
HbO2_absorption_coefficients = [290, 390, 518, 586, 650, 816, 916, 1022, 1058, 1154]
Hb_absorption_coefficients = [1794.28, 1102.2, 1405.24, 1548.52, 1311.88, 761.72, 693.76, 692.36, 691.32, 726.44 ]

absorption_list = [x for x in itertools.chain.from_iterable(itertools.zip_longest(HbO2_absorption_coefficients,Hb_absorption_coefficients)) if x]

########################################################################################################################
# Now implementing LU

# Function used to minimise the absolute difference between the simulated initial pressures and the spectra predicted from LU
def predictedp0_vs_simulated(coefficients, absorption_coefficients_and_simulated_spectra):
    aHgO2 = coefficients[0]
    aHg = coefficients[1]

    number_of_mua = int(2*len(absorption_coefficients_and_simulated_spectra)/3)
    absorption_coefficients = [absorption_coefficients_and_simulated_spectra[i] for i in range(number_of_mua)]
    simulated_spectra = [absorption_coefficients_and_simulated_spectra[i] for i in range(number_of_mua,len(absorption_coefficients_and_simulated_spectra))]

    predicted_spectra = [aHgO2 * absorption_coefficients[2*i] + aHg * absorption_coefficients[2*i + 1] for i in range(int(0.5 * len(absorption_coefficients)))]
    return np.linalg.norm(list(map(operator.sub, predicted_spectra, simulated_spectra)))

# Data for the PA reconstruction
f = h5py.File('I:/research/seblab/data/group_folders/Janek/kevindata/Scan_108.hdf5','r')
reconstructed_data = f['recons']['Backprojection Preclinical']['0']

# Gas Challenge parameters
timesteps = 93
measured_wavelengths = [700, 730, 750, 760, 770, 800, 820, 840, 850, 880]

# Plotting the segmentation lines for the tumour and reference region
s1 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines1.npy')
s2 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines2.npy')
s3 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines3.npy')
s4 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines4.npy')
s5 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines5.npy')
s6 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines6.npy')
s7 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines7.npy')
s8 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines8.npy')
s9 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines9.npy')
s10 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines10.npy')
t1 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/tumouroutlines1a.npy')
t2 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/tumouroutlines1b.npy')
# Collecting the coordinates of the points within the segmented regions
chosen_pixel_coords = np.load('../Datasets/GasChallengeFullMouse/Segmentation/pixel_coords.npy')
chosen_pixel_coords = list(chosen_pixel_coords)
chosen_pixel_coords = [list(entry) for entry in chosen_pixel_coords]
print(len(chosen_pixel_coords))
tumour_coords = np.load('../Datasets/GasChallengeFullMouse/Segmentation/tumourcoords2.npy')
tumour_coords = list(tumour_coords)
tumour_coords = [list(entry) for entry in tumour_coords]

tumour_indices = []
for i in range(len(tumour_coords)):
    tumour_indices.append(chosen_pixel_coords.index(tumour_coords[i]))
'''
# For each timestep, the model estimates so2 for each pixel:

so2_timeseries_bypixel = [[] for i in range(len(chosen_pixel_coords))]

for i in range(timesteps):
    print(i)
    filename = '../Datasets/GasChallengeFullMouse/Timestep' + str(i) + '.pt'
    f = torch.load(filename)
    initial_guess = [0.5, 0.5]
    bnds = ((0, None), (0, None))
    timestep_predictions = []
    for j in range(len(f)):
        initial_pressures = list(f[j])

        # Tuple of the absorption coefficients and the initial pressure spectrum which are arguments in the function
        func_input = absorption_list + initial_pressures
        # Calling minimisation function
        min = minimize(predictedp0_vs_simulated, initial_guess, args=func_input,bounds=bnds,  options= {'eps':0.01})
        timestep_predictions.append(min.x[0]/(min.x[0]+min.x[1]))
    
        print(min.x[0], min.x[1])
        
        print(initial_pressures)
        predicted_spectrum = [min.x[0] * absorption_list[2*i] + min.x[1] * absorption_list[2*i + 1] for i in range(int(0.5 * len(absorption_list)))]
        print(predicted_spectrum)
        print(np.linalg.norm(list(map(operator.sub, predicted_spectrum, initial_pressures))))
        
    for k in range(len(chosen_pixel_coords)):
        so2_timeseries_bypixel[k].append(np.float64(timestep_predictions[k]))

#np.save('../GasChallengeFullMouse_results/LU/so2_timeseries_bypixel.npy', np.array(so2_timeseries_bypixel)
'''
so2_timeseries_bypixel = np.load('../GasChallengeFullMouse_results/LU/so2_timeseries_bypixel.npy')
'''
# Plotting the results
for i in range(timesteps):
    background_image_data = reconstructed_data[i][5] # Background greyscale PA image at 800nm
    plt.imshow(background_image_data,cmap='gray',origin = 'lower')

    so2_data_fortimestep = [pixel[i] for pixel in so2_timeseries_bypixel]
    so2_map = [[np.NaN for i in range(250)] for j in range(250)]

    count = 0
    for entry in chosen_pixel_coords:
        x_coord = entry[0]
        y_coord = entry[1]
        so2_map[y_coord][x_coord] = so2_data_fortimestep[count]*100
        count += 1

    plt.imshow(so2_map,interpolation = 'nearest', vmin = 0, vmax = 100, origin = 'lower')
    clb = plt.colorbar()
    clb.ax.set_title('sO$_2$ [%]')

    file = '../GasChallengeFullMouse_results/LU/Absoluteso2maps/GasChallengeFullMouse_Timestep' + str(i)
    #plt.savefig(file)
    plt.show()
'''
# Now implementing a 'baseline so2 comparison'
N_baseline_timesteps = 46

# Time-averaged absolute so2 plot after baseline:

absolute_time_average_afterbaseline = [np.mean(pixeldata[N_baseline_timesteps:]) for pixeldata in so2_timeseries_bypixel]

background_image_data = reconstructed_data[0][5] # Background greyscale PA image at 800nm
plt.imshow(background_image_data,cmap='gray',origin = 'lower')
so2_map = [[np.NaN for i in range(250)] for j in range(250)]

count = 0
for entry in chosen_pixel_coords:
    x_coord = entry[0]
    y_coord = entry[1]
    so2_map[y_coord][x_coord] = absolute_time_average_afterbaseline[count] *100
    count += 1

plt.imshow(so2_map,interpolation = 'nearest', vmin = 0, vmax = 100, origin = 'lower')
clb = plt.colorbar()
clb.ax.set_title('sO$_2$ [%]')
plt.plot(*zip(*s1), color='k')
plt.plot(*zip(*s2), color='k')
plt.plot(*zip(*s3), color='k')
plt.plot(*zip(*s4), color='k')
plt.plot(*zip(*s5), color='k')
plt.plot(*zip(*s6), color='k')
plt.plot(*zip(*s7), color='k')
plt.plot(*zip(*s8), color='k')
plt.plot(*zip(*s9), color='k')
plt.plot(*zip(*s10), color='k')
plt.plot(*zip(*t1), color = 'k')
plt.plot(*zip(*t2), color = 'k')
file = '../GasChallengeFullMouse_results/LU/time_averaged_absolute.png'
plt.savefig(file)
plt.show()

pixel_baselines = []
for i in range(len(so2_timeseries_bypixel)):
    baseline = np.mean(so2_timeseries_bypixel[i][0:N_baseline_timesteps])
    pixel_baselines.append(baseline)

delta_so2_timeseries_bypixel = []
for i in range(len(so2_timeseries_bypixel)):
    delta_so2_series = list(np.array(so2_timeseries_bypixel[i]) - pixel_baselines[i])
    delta_so2_series = delta_so2_series[N_baseline_timesteps:]
    delta_so2_timeseries_bypixel.append(delta_so2_series)

#np.save('../GasChallengeFullMouse_results/LU/Baseline46_delta_so2_timeseries_bypixel.npy', np.array(delta_so2_timeseries_bypixel))


'''
# Plotting the delta_so2_maps
for i in range(timesteps-N_baseline_timesteps):
    background_image_data = reconstructed_data[i+N_baseline_timesteps][5] # Background greyscale PA image at 800nm


    delta_so2_data_fortimestep = [pixel[i] for pixel in delta_so2_timeseries_bypixel]

    delta_so2_map = [[np.NaN for i in range(250)] for j in range(250)]

    count = 0
    for entry in chosen_pixel_coords:
        x_coord = entry[0]
        y_coord = entry[1]
        delta_so2_map[y_coord][x_coord] = delta_so2_data_fortimestep[count]*100
        count += 1
    fig,axes = plt.subplots()
    plt.imshow(background_image_data, cmap='gray', origin='lower')
    plt.imshow(delta_so2_map,interpolation = 'nearest', vmin = -70, vmax = 70, origin = 'lower', cmap=cmap)
    clb = plt.colorbar()
    clb.ax.set_title('\u0394sO$_2$ [%]')
    plt.plot(*zip(*s1), color='r')
    plt.plot(*zip(*s2), color='r')
    plt.plot(*zip(*s3), color='r')
    plt.plot(*zip(*s4), color='r')
    plt.plot(*zip(*s5), color='r')
    plt.plot(*zip(*s6), color='r')
    plt.plot(*zip(*s7), color='r')
    plt.plot(*zip(*s8), color='r')
    plt.plot(*zip(*s9), color='r')
    plt.plot(*zip(*s10), color='r')
    file = '../GasChallengeFullMouse_results/LU/Deltaso2maps/GasChallengeFullMouse_deltaso2_baseline46_Timestep' + str(i+N_baseline_timesteps)
    #plt.savefig(file)
    plt.show()
'''

# Creating a histogram of the timeaveraged absolute change in so2 for each pixel
time_averaged_delta_so2_bypixel =  [np.mean(entry) for entry in delta_so2_timeseries_bypixel]
time_averaged_absolute_delta_so2_bypixel = [np.mean(abs(np.array(entry))) for entry in delta_so2_timeseries_bypixel]

time_averaged_delta_so2_tumourpixels = []
time_averaged_delta_so2_nontumourpixels = []
time_averaged_absolute_delta_so2_tumourpixels = []
time_averaged_absolute_delta_so2_nontumourpixels = []
for i in range(len(time_averaged_absolute_delta_so2_bypixel)):
    if i in tumour_indices:
        time_averaged_absolute_delta_so2_tumourpixels.append(time_averaged_absolute_delta_so2_bypixel[i])
        time_averaged_delta_so2_tumourpixels.append(time_averaged_delta_so2_bypixel[i])
    else:
        time_averaged_absolute_delta_so2_nontumourpixels.append(time_averaged_absolute_delta_so2_bypixel[i])
        time_averaged_delta_so2_nontumourpixels.append(time_averaged_delta_so2_bypixel[i])


background_image_data = reconstructed_data[0][5] # Background greyscale PA image at 800nm
delta_so2_map = [[np.NaN for i in range(250)] for j in range(250)]

count = 0
for entry in chosen_pixel_coords:
    x_coord = entry[0]
    y_coord = entry[1]
    delta_so2_map[y_coord][x_coord] =time_averaged_delta_so2_bypixel[count] *100
    count += 1

fig, axes = plt.subplots()
plt.imshow(background_image_data, cmap='gray', origin='lower')
plt.imshow(delta_so2_map, interpolation='nearest', vmin=-70, vmax=70, origin='lower', cmap=cmap)
clb = plt.colorbar()
clb.ax.set_title('\u0394sO$_2$ [%]')
plt.plot(*zip(*s1), color='k')
plt.plot(*zip(*s2), color='k')
plt.plot(*zip(*s3), color='k')
plt.plot(*zip(*s4), color='k')
plt.plot(*zip(*s5), color='k')
plt.plot(*zip(*s6), color='k')
plt.plot(*zip(*s7), color='k')
plt.plot(*zip(*s8), color='k')
plt.plot(*zip(*s9), color='k')
plt.plot(*zip(*s10), color='k')
plt.plot(*zip(*t1), color = 'k')
plt.plot(*zip(*t2), color = 'k')
file = '../GasChallengeFullMouse_results/LU/time_averaged_delta.png'
plt.savefig(file)
plt.show()

plt.hist(np.array(time_averaged_delta_so2_tumourpixels)*100, bins = 100, range = [-50,50], facecolor = 'gray', align = 'mid')
plt.xlabel('Time-averaged \u0394sO$_2$ [%] ')
plt.ylabel('Frequency')
plt.savefig('../GasChallengeFullMouse_results/LU/Baseline46_TumourHistogram0.png')
plt.show()

plt.hist(np.array(time_averaged_delta_so2_nontumourpixels) * 100, bins=100, range=[-50, 50], facecolor='gray', align='mid')
plt.xlabel('Time-averaged \u0394sO$_2$ [%] ')
plt.ylabel('Frequency')
plt.savefig('../GasChallengeFullMouse_results/LU/Baseline46_NonTumourHistogram0.png')
plt.show()

plt.hist(np.array(time_averaged_absolute_delta_so2_tumourpixels)*100, bins = 100, range = [0,50], facecolor = 'gray', align = 'mid')
plt.xlabel('Time-averaged |\u0394sO$_2$| [%] ')
plt.ylabel('Frequency')
plt.savefig('../GasChallengeFullMouse_results/LU/Baseline46_TumourHistogram.png')
plt.show()
plt.hist(np.array(time_averaged_absolute_delta_so2_nontumourpixels)*100, bins = 100, range = [0,50], facecolor = 'gray', align = 'mid')
plt.xlabel('Time-averaged |\u0394sO$_2$| [%] ')
plt.ylabel('Frequency')
plt.savefig('../GasChallengeFullMouse_results/LU/Baseline46_NonTumourHistogram.png')
plt.show()